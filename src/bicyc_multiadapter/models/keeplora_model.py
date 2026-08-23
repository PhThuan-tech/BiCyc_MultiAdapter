"""Direction-1 assembled model: frozen ViT + per-task KeepLoRA factors.

Lifecycle of one task t (see docs/DIRECTION1_SPEC.md):
  1. ``expand_head``  - grow the linear training head for the new classes.
  2. ``begin_task``   - one CE-only backward pass yields G_t per target layer;
                        residual-gradient SVD initializes frozen A_t / trainable B_t;
                        forward hooks start caching layer inputs.
  3. training epochs  - KeepLoRATrainer steps; PFD means updated online per batch.
  4. ``end_task``     - compact feature memory M_t update, freeze (and optionally
                        merge = original KeepLoRA behaviour).
  5. ``snapshot``     - immutable teacher for the next task.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .adapters.keeplora import (
    KeepLoRAFactors,
    initialize_lora_from_gradient,
    orthonormal_union,
    principal_weight_subspace,
    update_feature_subspace,
)
from .adapters.routing import RoutedKeepLoRALinear
from .backbones.frozen_encoder import FrozenFeatureEncoder

DEFAULT_TARGET_PATTERNS = ("qkv", "proj", "fc1", "fc2")


@dataclass
class LayerMemory:
    """Per-layer compact state; only bases are checkpointed, never raw activations."""

    weight_basis: Tensor | None = None  # W_p [d_in, p]
    feature_basis: Tensor | None = None  # M_t [d_in, m], kept on CPU
    activation_cache: list[Tensor] = field(default_factory=list)  # CPU chunks
    cached_rows: int = 0
    last_input: Tensor | None = None  # latest batch (device) for online PFD mean
    hook_handle: object | None = None


class KeepLoRACILModel(nn.Module):
    """Frozen encoder whose selected linears carry one KeepLoRA factor per task.

    ``merge_after_task=True`` reproduces the original KeepLoRA baseline (delta is
    folded back into W at end of task); ``False`` keeps frozen factors in the
    bank for PFD routing (the proposed hybrid).
    """

    def __init__(
        self,
        encoder: FrozenFeatureEncoder,
        feature_dim: int,
        rank: int = 8,
        alpha: float = 16.0,
        weight_energy: float = 0.95,
        feature_energy: float = 0.95,
        merge_after_task: bool = False,
        router_similarity: str = "l2",
        router_temperature: float = 1.0,
        router_top_k: int | None = None,
        target_patterns: tuple[str, ...] = DEFAULT_TARGET_PATTERNS,
        activation_cache_rows: int = 4096,
    ) -> None:
        super().__init__()
        self.encoder = encoder  # already frozen inside its constructor
        self.feature_dim = feature_dim
        self.rank, self.alpha = rank, alpha
        self.weight_energy, self.feature_energy = weight_energy, feature_energy
        self.merge_after_task = merge_after_task
        self.target_patterns = tuple(target_patterns)
        self.activation_cache_rows = activation_cache_rows
        self.router_similarity = router_similarity
        self.router_temperature = router_temperature
        self.router_top_k = router_top_k
        self.layers = nn.ModuleDict()  # name -> patched linear
        self.memory: dict[str, LayerMemory] = {}
        self.head: nn.Linear | None = None

    def patch_backbone(self) -> None:
        """Swap selected ``nn.Linear`` blocks for :class:`RoutedKeepLoRALinear` (once).

        PyTorch stores weights as [d_out, d_in]; the KeepLoRA convention here is
        ``y = x @ W`` so the weight is transposed at patch time.
        """
        for name, module in list(self.encoder.network.named_modules()):
            if not isinstance(module, nn.Linear) or not name.endswith(self.target_patterns):
                continue
            parent_name, _, leaf = name.rpartition(".")
            parent = self.encoder.network.get_submodule(parent_name) if parent_name else self.encoder.network
            wrapped = RoutedKeepLoRALinear(
                module.weight.t().detach(), module.bias, self.alpha, self.router_similarity, self.router_temperature
            )
            wrapped.top_k = self.router_top_k
            setattr(parent, leaf, wrapped)
            # nn.ModuleDict forbids dots in keys, so flatten the module path.
            registry_key = name.replace(".", "-")
            memory = LayerMemory()
            # Principal weight basis W_p computed once and reused by every task.
            memory.weight_basis = principal_weight_subspace(wrapped.base_weight, self.weight_energy)
            self.layers[registry_key] = wrapped
            self.memory[registry_key] = memory
        if not self.layers:
            raise RuntimeError("No target linear layer matched; check ``target_patterns``.")

    def expand_head(self, class_ids: tuple[int, ...]) -> None:
        """Grow the training head; old rows copied verbatim, new rows freshly init.

        The head is created lazily, i.e. after ``Model.to(device)`` has already
        run, so every newly created/shrunk head must be moved back to the device
        inferred from the patched layers' frozen base weights.
        """
        width = max(class_ids) + 1
        device = None
        if self.head is not None:
            device = self.head.weight.device
        elif self.layers:
            device = next(iter(self.layers.values())).base_weight.device
        if self.head is None:
            self.head = nn.Linear(self.feature_dim, width)
        else:
            if width <= self.head.out_features:
                return
            widened = nn.Linear(self.feature_dim, width)
            old_width = self.head.out_features
            with torch.no_grad():
                widened.weight[:old_width] = self.head.weight
                widened.bias[:old_width] = self.head.bias
                nn.init.kaiming_uniform_(widened.weight[old_width:], a=5**0.5)
                widened.bias[old_width:].zero_()
            self.head = widened
        if device is not None:
            self.head = self.head.to(device)

    def begin_task(self, task_id: int, train_loader, device) -> None:
        """Residual-gradient SVD init from a single CE-only pass (KeepLoRA Eq. 6)."""
        self._set_base_grad_enabled(True)
        sample_count = self._accumulate_classification_gradient(train_loader, device)
        with torch.no_grad():
            for name, layer in self.layers.items():
                memory = self.memory[name]
                gradient = layer.base_weight.grad / sample_count  # G_t, mean over stream
                protected = orthonormal_union(
                    memory.weight_basis.to(device),
                    None if memory.feature_basis is None else memory.feature_basis.to(device),
                )
                factors: KeepLoRAFactors = initialize_lora_from_gradient(gradient, protected, self.rank)
                layer.add_task(task_id, factors)
                layer.base_weight.grad = None
        self._set_base_grad_enabled(False)
        self._register_hooks()

    def _accumulate_classification_gradient(self, train_loader, device) -> int:
        """Backward CE over the whole task stream; only base weights/head collect grads.

        The gradient w.r.t. ``base_weight`` flows through the base path only (the
        LoRA delta does not depend on W), so this is exactly G_t of the paper.
        """
        seen = 0
        for images, labels in train_loader:
            logits, _ = self.forward(images.to(device, non_blocking=True))
            F.cross_entropy(logits, labels.to(device)).backward()
            seen += labels.shape[0]
        return seen

    def _set_base_grad_enabled(self, enabled: bool) -> None:
        """Temporarily make base weights leaves that collect the init gradient."""
        for layer in self.layers.values():
            if enabled:
                layer.base_weight.grad = None
            layer.base_weight.requires_grad_(enabled)

    def _register_hooks(self) -> None:
        """Capture layer inputs: last batch feeds PFD means; capped CPU cache feeds the M_t SVD."""
        for name, layer in self.layers.items():

            def hook(_module, inputs, _output, key=name):
                detached = inputs[0].detach()
                # Transformer linears see [batch, tokens, d_in]; PFD statistics and the
                # end-of-task SVD both operate on flat [rows, d_in] activation rows.
                flat = detached.reshape(-1, detached.shape[-1])
                memory = self.memory[key]
                memory.last_input = flat
                remaining = self.activation_cache_rows - memory.cached_rows
                if remaining > 0:
                    kept = flat[:remaining]
                    memory.activation_cache.append(kept.to("cpu"))
                    memory.cached_rows += kept.shape[0]

            self.memory[name].hook_handle = layer.register_forward_hook(hook)

    @torch.no_grad()
    def update_routing_statistics(self, task_id: int) -> None:
        """Online PFD update D_t^l = E[W^l h^l(x)] from the most recent batch (Eq. 3 of PFD)."""
        for name, layer in self.layers.items():
            last = self.memory[name].last_input
            if last is not None:
                layer.update_distribution(task_id, last)

    def trainable_parameters(self) -> list[Tensor]:
        """Model-optimizer scope: current-task B factors plus the growing head."""
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def end_task(self, task_id: int) -> None:
        """Refresh compact feature memory, freeze the task factor, optionally merge."""
        for name, layer in self.layers.items():
            memory = self.memory[name]
            if memory.activation_cache:
                # ``.float()`` guards the SVD against half-precision caches (AMP runs).
                stacked = torch.cat(memory.activation_cache, dim=0).float()  # [N, d_in] on CPU
                # M_t = orth([M_{t-1}, dominant residual input directions]).
                memory.feature_basis = update_feature_subspace(
                    stacked, memory.weight_basis, memory.feature_basis, self.feature_energy
                )
            if memory.hook_handle is not None:
                memory.hook_handle.remove()
                memory.hook_handle = None
            memory.activation_cache, memory.cached_rows, memory.last_input = [], 0, None
            layer.freeze_task(task_id)
            if self.merge_after_task:
                layer.merge_task(task_id)  # original KeepLoRA: W <- W + scaling*A*(B-B0)

    def snapshot(self) -> "KeepLoRACILModel":
        """Deep-copied teacher; old features come from current-task images only (EFCIL-safe)."""
        clone = copy.deepcopy(self)
        clone.eval()
        for parameter in clone.parameters():
            parameter.requires_grad_(False)
        return clone

    # ------------------------------------------------------- checkpoint helpers
    def memory_state_dict(self) -> dict[str, dict[str, Tensor | None]]:
        """Compact per-layer bases for checkpoints; raw activations are never saved."""
        return {
            name: {
                "weight_basis": None if memory.weight_basis is None else memory.weight_basis.detach().cpu(),
                "feature_basis": None if memory.feature_basis is None else memory.feature_basis.detach().cpu(),
            }
            for name, memory in self.memory.items()
        }

    def load_memory_state(self, state: dict[str, dict[str, Tensor | None]]) -> None:
        """Restore protected bases (W_p, M_t) saved by :meth:`memory_state_dict`."""
        for name, entry in state.items():
            if name in self.memory:
                self.memory[name].weight_basis = entry.get("weight_basis")
                self.memory[name].feature_basis = entry.get("feature_basis")

    @torch.no_grad()
    def trim_structure(self, state: dict) -> None:
        """Drop adapters/PFD means (and shrink the head) absent from ``state``.

        Together with :meth:`restore_structure` this lets any checkpoint payload
        be applied to the current model, even an older one with less structure.
        """
        for layer in self.layers.values():
            wanted_tasks: set[int] = set()
            wanted_means: set[int] = set()
            for key in state:
                if ".adapters." in key and key.endswith(".B"):
                    raw_registry, remainder = key.split(".adapters.")
                    if self._resolve_registry(raw_registry) is layer:
                        wanted_tasks.add(int(remainder.split(".")[0]))
                elif ".router.means." in key:
                    raw_registry, task_key = key.split(".router.means.")
                    if self._resolve_registry(raw_registry) is layer:
                        wanted_means.add(int(task_key))
            for task_id in [int(k) for k in layer.adapters if int(k) not in wanted_tasks]:
                layer.adapters.pop(str(task_id))
            for task_key in [k for k in layer.router.means if int(k) not in wanted_means]:
                layer.router.means.pop(task_key)
                layer.router.counts.pop(task_key)
        if "head.weight" in state and self.head is not None:
            width = state["head.weight"].shape[0]
            if self.head.out_features > width:
                device = self.head.weight.device
                shrunk = nn.Linear(self.head.in_features, width)
                shrunk.weight.copy_(self.head.weight[:width])
                shrunk.bias.copy_(self.head.bias[:width])
                self.head = shrunk.to(device)

    def _resolve_registry(self, name: str):
        """State keys may reference a layer via ``encoder.network.*`` or the flat dict key."""
        if ".network." in name:
            name = name.split(".network.", 1)[1]  # drop the encoder module prefix
        for candidate in (name, name.replace(".", "-")):
            if candidate in self.layers:
                return self.layers[candidate]
        return None

    @torch.no_grad()
    def restore_structure(self, state: dict) -> None:
        """Recreate dynamic modules (adapters, PFD means, head) to match ``state``.

        Adapters and router statistics live in ``nn.ModuleDict``/``nn.ParameterDict``
        members that are created at runtime, so a freshly patched model cannot
        ``load_state_dict`` a checkpoint without rebuilding them first. Shapes are
        taken from the checkpoint tensors themselves, which also supports the
        variable effective rank produced by the residual-SVD initialization.
        """
        highest_task = -1
        head_width = 0
        for key, value in state.items():
            if ".adapters." in key and key.endswith(".B"):
                raw_registry, remainder = key.split(".adapters.")
                layer = self._resolve_registry(raw_registry)
                if layer is None:
                    continue
                task_id = int(remainder.split(".")[0])
                if str(task_id) not in layer.adapters:
                    prefix = f"{key[: -len('.B')]}."
                    factors = KeepLoRAFactors(
                        A=state[prefix + "A"].detach().clone(),
                        B=value.detach().clone(),
                        projected_gradient=torch.zeros(1),
                    )
                    layer.add_task(task_id, factors)
                highest_task = max(highest_task, task_id)
            elif ".router.means." in key:
                raw_registry, task_key = key.split(".router.means.")
                layer = self._resolve_registry(raw_registry)
                if layer is not None and task_key not in layer.router.means:
                    layer.router.means[task_key] = nn.Parameter(value.detach().clone(), requires_grad=False)
            elif ".router.counts." in key:
                raw_registry, task_key = key.split(".router.counts.")
                layer = self._resolve_registry(raw_registry)
                if layer is not None and task_key not in layer.router.counts:
                    layer.router.counts[task_key] = nn.Parameter(value.detach().clone(), requires_grad=False)
            elif key == "head.weight":
                head_width = value.shape[0]
        if head_width and (self.head is None or self.head.out_features < head_width):
            self.expand_head(tuple(range(head_width)))
        # Only the newest task stays trainable; earlier factors are frozen history.
        for layer in self.layers.values():
            for task_id in range(highest_task):
                if str(task_id) in layer.adapters:
                    layer.freeze_task(task_id)

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor]:
        """Return (logits from the growing head, pooled backbone features)."""
        features = self.encoder.forward_features(images)
        logits = self.head(features)
        return logits, features
