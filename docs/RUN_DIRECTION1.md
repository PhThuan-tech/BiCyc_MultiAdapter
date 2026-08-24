# Hướng dẫn chạy Hướng 1 — KeepLoRA + BiCyc + PFD routing + Adaptive Gate

Tài liệu này mô tả cách cài đặt môi trường, chuẩn bị dữ liệu và chạy baseline/ablation cho
[Hướng 1](DIRECTION1_SPEC.md): **KeepLoRA nhiều adapter + BiCyc alignment + PFD routing +
adaptive Gaussian-KL gate** (đề xuất của nhóm).

## 1. Kiến trúc & luồng một task

```text
image ──► Frozen ViT (timm) ──► [qkv | proj | fc1 | fc2] đã vá RoutedKeepLoRALinear
                                        │  (delta = α/r · A_t(B_t−B_t⁰), A_t frozen)
                                        ▼
                              pooled feature z_new ──► linear head (train CE)
                                        │
        old snapshot f_{t-1}(x) ──► BiCyc A/D maps (2 optimizer tách biệt)
                                        │
                                        ▼
                    Gaussian CIL classifier (μ_c, Σ_c); class cũ vận chuyển qua A
```

Vòng đời mỗi task `t` trong `DirectionOneExperiment.run()`:

1. **`expand_head`** — mở rộng head linear cho class mới (row cũ giữ nguyên).
2. **`begin_task`** — chạy **một** backward CE-only trên toàn bộ stream của task để lấy gradient
   `G_t` từng layer → khởi tạo `(A_t, B_t)` bằng SVD residual trên basis bảo vệ
   `Q = orth([W_p, M_{t-1}])`; đồng thời gắn forward hook thu input activation (cache CPU, có cap).
3. **Train epochs** — `KeepLoRATrainer.train_batch` gồm 2 bước tách biệt:
   - *Model step*: `CE + λ_t·λ_bi·‖D(z_new) − sg(z_old)‖²` cập nhật `B_t` + head;
     `λ_t = λ_min + (λ_max−λ_min)·exp(−δ_t/τ)` với `δ_t` là symmetric diagonal Gaussian-KL
     (tắt bằng `alignment.adaptive_gate=false` để cố định `λ_t=1`).
   - *Alignment step*: toàn bộ `L_BiCyc = λ_bi·L_bi + λ_cyc·L_cyc` trên feature đã detach,
     chỉ học map A/D, không bao giờ truyền gradient về `B_t`.
   - Sau mỗi batch gọi `update_routing_statistics` để cập nhật online mean PFD
     `D_t^l = E[W^l h^l(x)]`.
4. **Consolidation** — vận chuyển thống kê class cũ qua map A (`μ'=Aμ`, `Σ'=AΣAᵀ`) rồi fit
   thống kê mới của task hiện tại cho Gaussian classifier.
5. **`end_task`** — cập nhật compact feature memory `M_t = orth([M_{t-1}, U_residual])`
   (energy threshold `feature_energy`), freeze `B_t`; nếu `merge_after_task=true` thì gộp delta
   về `W` (KeepLoRA nguyên gốc).
6. **`snapshot`** — deep-copy teacher cho task kế tiếp (chỉ dùng ảnh của task hiện tại,
   hợp lệ exemplar-free).


## 2. Yêu cầu môi trường

| Thành phần | Yêu cầu |
| --- | --- |
| OS | Linux / Windows / macOS |
| Python | **3.11 hoặc 3.12** (`requires-python = ">=3.11,<3.13"`) |
| GPU | Khuyến nghị ≥ 12 GB VRAM (batch 128); GPU 8 GB (vd. RTX 4060 Laptop) chạy được với preset `*_8gb` — xem mục 2.3 |
| CUDA | Driver ≥ CUDA 12.4 nếu dùng wheel GPU `torch==2.5.1` |
| Thư viện chính | xem `requirements/base.txt`: hydra-core, timm, torchvision (cài kèm torch), datasets, tensorboard, tqdm… |

### 2.1 Cài đặt cục bộ

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS:       source .venv/bin/activate
python -m pip install --upgrade pip

# GPU CUDA 12.4 (torchvision đi kèm để tải CIFAR-100)
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
# CPU-only: pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements/base.txt
pip install -e .
```

Kiểm tra nhanh unit test:

```bash
pip install pytest==8.3.4
pytest tests/unit -q
```

Kiểm tra vòng đời model không cần GPU/timm (encoder giả, chạy vài giây):

```bash
python scripts/smoke_direction1.py   # mong đợi: "SMOKE TEST OK"
```

### 2.2 Docker (khuyến nghị cho GPU)

```bash
docker compose build
docker compose run --rm research bash
# bên trong container:
python -m bicyc_multiadapter.train experiment=keeplora_bicyc
```

Image nền `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`; `data/`, `outputs/`, `checkpoints/`
là host volume nên dữ liệu/checkpoint không bị đóng vào image.

### 2.3 Cấu hình cho máy local: RTX 4060 Laptop 8 GB

Thông số tham chiếu: `NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB, driver 576.57 (CUDA 12.9)`.
Driver CUDA 12.9 **cao hơn** runtime 12.4 nên dùng trực tiếp wheel `cu124`, không cần cài CUDA Toolkit.

```powershell
# 1) Tạo môi trường mới (không tái sử dụng env đang có torch bản CPU)
conda create -n bicyc python=3.11 -y
conda activate bicyc
#    hoặc: python -m venv .venv ; .venv\Scripts\Activate.ps1

# 2) PyTorch GPU + torchvision (bắt buộc cùng index để khớp phiên bản)
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

# 3) Thư viện còn lại + project
pip install -r requirements/base.txt
pip install -e .

# 4) Xác nhận GPU được nhận
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
#    mong đợi: 2.5.1+cu124 True NVIDIA GeForce RTX 4060 Laptop GPU

# 5) Kiểm tra logic nhanh trước khi train thật
python scripts/smoke_direction1.py   # mong đợi: "SMOKE TEST OK"
```

> ⚠️ Nếu `import torch` in ra `2.12.1+cpu` (như env hiện tại của máy tham chiếu) thì đó là bản
> **CPU-only** — phải cài lại bằng lệnh bước 2 trong môi trường mới, không trộn hai wheel.

Chạy với preset đã tối ưu cho 8 GB VRAM (batch 32, cache activation nhỏ):

```powershell
# Đề xuất đầy đủ (5): routed multi-adapter + BiCyc + adaptive gate
python -m bicyc_multiadapter.train experiment=keeplora_bicyc_8gb

# Baseline KeepLoRA nguyên gốc (1)
python -m bicyc_multiadapter.train experiment=keeplora_original_8gb

# Ablation (4): bỏ adaptive gate
python -m bicyc_multiadapter.train experiment=keeplora_bicyc_8gb alignment.adaptive_gate=false
```

Ngân sách VRAM 8 GB với preset này (ViT-B/16 @224, fp32):

| Thành phần | Ước lượng |
| --- | --- |
| Backbone frozen (+ snapshot teacher) | ~0.85 GB |
| Activations forward/backward, batch 32 | ~3.5–4.5 GB |
| BiCyc maps, head, optimizer states, SVD init | ~0.3 GB |
| **Tổng** | **~5–6 GB** (desktop/Windows đã chiếm ~0.7 GB trên card) |

Nếu vẫn thiếu VRAM, giảm dần theo thứ tự:

1. `experiment.train.batch_size=16`
2. `experiment.activation_cache_rows=1024` (chỉ ảnh hưởng RAM CPU, không phải VRAM)
3. `experiment.targets=[qkv,proj]` — giữ 2/4 loại projection (ảnh hưởng phạm vi thuật toán, ghi nhận khi báo cáo)
4. `experiment.image_size=168` — thay đổi protocol, chỉ dùng cho thử nghiệm nhanh

### 2.4 Smoke test toàn bộ pipeline (10 tasks, tránh OOM)

Trước khi chạy thật (nhiều giờ), hãy xác minh pipeline chạy đủ **10 tasks** trên đúng GPU của bạn
mà **không OOM**. Trên Colab chỉ cần bật `SMOKE_TEST = True` ở cell 5 (Tham số run) — notebook tự
ghim batch 16 + 1 epoch/task + cache nhỏ (peak VRAM ~4GB trên T4). Chạy CLI tương đương:

```bash
python -m bicyc_multiadapter.train experiment=keeplora_bicyc_8gb \
    experiment.seed=2024 \
    experiment.data.root=/content/data/cifar100 \
    experiment.data.num_workers=2 \
    output_dir=/content/outputs/keeplora_bicyc_8gb/smoke \
    experiment.train.epochs_per_task=1 \
    experiment.train.batch_size=16 \
    experiment.activation_cache_rows=1024 \
    experiment.checkpoint_every_epochs=0
```

Kết quả đạt là run kết thúc với `Run hoan tat...` trong `run.log`, đủ
`run_meta.json`/`train_log.csv`/`metrics.json`, và dòng task summary cuối có
`GPU peak` thấp hơn hẳn VRAM card (vd. peak ~4GB trên T4 16GB → an toàn). Nếu `GPU peak`
vượt ~85% VRAM, giảm `batch_size` (32 → 16 → 8) trước khi chạy thật.

> ⚠️ Smoke test **không có ý nghĩa thống kê** (1 epoch/task). Đây chỉ là kiểm tra pipeline +
> đo VRAM peak.

## 3. Dữ liệu

- **CIFAR-100** tự động tải về `data/cifar100/` lần chạy đầu (qua torchvision, cần Internet).
- Chia task: 10 classes đầu + 9×10 classes, class order sinh từ seed `1993`
  (`configs/data/cifar100_10tasks.yaml`). Không lưu bất kỳ raw sample cũ nào.

## 4. Chạy thí nghiệm

Chạy từ thư mục gốc repo. Output ghi vào `outputs/<experiment.name>/seed_<seed>/`.

```bash
# (5) Đề xuất đầy đủ: routed multi-adapter + BiCyc + adaptive gate
python -m bicyc_multiadapter.train experiment=keeplora_bicyc

# Bật mixed precision fp16 (GPU TensorCore: T4/RTX/A100) — tăng tốc ~1.5-2x
python -m bicyc_multiadapter.train experiment=keeplora_bicyc experiment.train.amp=true

# (1) Baseline KeepLoRA nguyên gốc: merge sau task, không distillation
python -m bicyc_multiadapter.train experiment=keeplora_original

# (4) Routed multi-adapter + BiCyc, bỏ adaptive gate (λ_t ≡ 1)
python -m bicyc_multiadapter.train experiment=keeplora_bicyc experiment.alignment.adaptive_gate=false

# (2)/(3) BiCyc cố định λ_t=1 với hệ số khác nhau:
python -m bicyc_multiadapter.train experiment=keeplora_bicyc experiment.alignment.adaptive_gate=false \
    experiment.alignment.lambda_bi=1.0 experiment.alignment.lambda_cyc=0.0

# Gộp adapter sau mỗi task nhưng giữ BiCyc (so sánh merge vs routed):
python -m bicyc_multiadapter.train experiment=keeplora_bicyc experiment.keeplora.merge_after_task=true

# Đổi seed (chạy ≥ 3 seeds cho mỗi cấu hình):
python -m bicyc_multiadapter.train experiment=keeplora_bicyc experiment.seed=0
```

Chạy lại đánh giá từ checkpoint (không train lại):

```bash
python -m bicyc_multiadapter.evaluate experiment=keeplora_bicyc
```

Theo dõi loss/metric qua TensorBoard:

```bash
tensorboard --logdir outputs
```

## 5. Kết quả đầu ra

Sau mỗi run, thư mục `outputs/<name>/seed_<seed>/` chứa:

| File | Nội dung |
| --- | --- |
| `run.log` | Log có timestamp (đầu run: bảng `RUN CONFIG` gồm experiment, seed, backbone, batch_size, epochs_per_task, lr/wd, AMP, targets, alignment λ, cache rows, data_root, output_dir). Sau mỗi task: `acc= <row> | delta_cu=<thay đổi acc các task cũ> | last_avg inc_avg forget | time cum eta | GPU alloc=...GiB peak=...GiB` — xem ngay forgetting từng bước + VRAM peak. |
| `run_meta.json` | Metadata môi trường (Python, torch, timm, CUDA device, git commit, config resolved) + summary cuối run |
| `config_resolved.yaml` | Bản dump toàn bộ cấu hình Hydra sau khi resolve — giúp tái tạo chính xác run |
| `checkpoint_last.pt` | state_dict model (basis W_p, M_t, factor A/B, PFD means), map BiCyc, thống kê Gaussian classifier, class order, accuracy matrix |
| `checkpoint_boundary.pt` | **Rolling** snapshot sau mỗi task hoàn tất (ghi đè file cũ) — mốc resume đầu task mới |
| `checkpoint_live.pt` | **Rolling** snapshot giữa task (mỗi `checkpoint_every_epochs` epoch hoặc khi Ctrl+C), kèm state 2 optimizer + RNG; tự xóa khi task xong |
| `checkpoint_task_<t>.pt` | (Tùy chọn, khi `experiment.keep_task_checkpoints=true`) Bản snapshot riêng cho từng task — tốn ~350 MB/task |
| `metrics.json` | `last_average`, `incremental_average`, `forgetting` + accuracy matrix đầy đủ (cập nhật sau mỗi task) |
| `accuracy_matrix.csv` | ma trận độ chính xác [task_eval × task_seen] |
| `history.jsonl` | mỗi task một dòng: per-task accuracy + `running_last_average/incremental_average/forgetting` tích lũy + `task_duration_seconds` + `train` (stat last epoch) |
| `train_log.csv` | loss (CE/model/backward/alignment, KL distance, λ_adaptive, phase_scale, samples_per_sec, epoch_time_s) theo từng epoch. Từ bản sửa gate: `distribution/distance` là KL thô (tổng theo chiều), `distribution/distance_per_dim` là KL trung bình/chiều — giá trị dùng để điều khiển gate nên nằm ở scale tương đương giữa các model có feature_dim khác nhau |
| `tensorboard/` | loss CE/model/alignment, KL distance, λ_adaptive theo từng epoch/task + `routing/own_weight/{layer}` (diagnostic) |
| `run.log` | dòng `[routing] eval task T -> <layer>: own=… max_other=…` sau mỗi task: tỉ lệ trọng số router rơi vào adapter đúng của task được eval. Nếu task cũ bị route sang adapter task mới (`own` thấp), đó là bằng chứng trực tiếp của routing interference gây forgetting |

### 5.1 Lưu ý về checkpoint

- **CPU-safe**: tất cả tensor được chuyển về CPU trước khi ghi file, nên việc lưu checkpoint không cần thêm GPU memory (quan trọng khi lưu ngay sau OOM).
- **Versioning**: checkpoint có trường `checkpoint_version` để đảm bảo tương thích ngược khi thay đổi schema.
- **Atomic write**: ghi qua file `.tmp` rồi rename, không bị corrupt nếu mất điện giữa chừng.

## 6. Resume khi bị ngắt giữa chừng (Colab/Kaggle/local)

Pipeline **tự động resume**: chỉ cần chạy lại đúng lệnh train với cùng `output_dir`.

- Sau mỗi task: ghi đè `checkpoint_boundary.pt`.
- Giữa task (`experiment.checkpoint_every_epochs > 0`, khuyến nghị = 1 trên cloud): ghi đè
  `checkpoint_live.pt` gồm model + cả hai optimizer + RNG state; ngoài ra bắt `KeyboardInterrupt`
  để lưu ngay khi bạn bấm stop.
- Lần chạy kế tiếp sẽ ưu tiên `checkpoint_live.pt` (resume đúng epoch giữa task; teacher cũ được
  dựng lại từ boundary gần nhất), nếu không có thì dùng `checkpoint_boundary.pt` (bắt đầu task kế).
- Chỉ tồn tại tối đa 1 checkpoint mỗi loại (file cũ bị xóa/ghi đè) — không phình dung lượng.
- Tắt resume bằng `experiment.resume=false`; lưu ý activation cache trong epoch dở dang không được
  lưu (chỉ ảnh hưởng mẫu SVD cuối task, số lượng vẫn đủ do cap cao hơn mức dùng thực tế).
## 7. Lưu ý quan trọng

- **Gradient isolation**: alignment optimizer chỉ thấy feature đã `detach()`; unit test
  `tests/unit/test_bicyc.py` kiểm chứng cycle loss không backprop về `z_new`.
- **Không lưu raw data cũ**: chỉ checkpoint basis trực chuẩn (`W_p`, `M_t`), mean PFD và
  `(μ_c, Σ_c)` — tuân thủ ràng buộc exemplar-free.
- **`activation_cache_rows`** (mặc định 4096) giới hạn số row activation giữ trên CPU per layer
  cho SVD cuối task (~48 layer × 4096 × 768 × 4B ≈ 600 MB RAM). Giảm nếu thiếu RAM.
- **Merge vs routed**: hai biến thể nghiên cứu bắt buộc phải ablation trực tiếp
  (xem DIRECTION1_SPEC.md, mục "Ablation tối thiểu").
- **Snapshot teacher** tăng thêm ~350 MB VRAM (deep-copy ViT-B); với GPU 8 GB hãy dùng preset
  `*_8gb.yaml` (mục 2.3).
- **Colab**:
  - **Python version**: Chọn runtime **Python 3.11 hoặc 3.12** (Runtime → Change runtime type).
    Python 3.13 không tương thích vì `numpy==1.26.4`, `scipy==1.14.1`, `scikit-learn==1.5.2` không có
    wheel cho 3.13. Notebook có guard tự động kiểm tra.
  - **Outputs**: Mặc định ghi vào đĩa local `/content/outputs/` (nhanh, tránh rate-limit Google Drive).
    Cuối run notebook tự động zip + copy sang Drive. Muốn resume qua các phiên, đặt
    `OUTPUTS_ON_DRIVE = True` trong cell 7 (Tham số run) — chậm hơn (~20–60s/lần checkpoint 350MB)
    nhưng giữ được checkpoint khi session chết.
  - **Data**: CIFAR-100 tự tải 1 lần, lưu vào Google Drive (không tải lại giữa các phiên).
  - **GPU**: T4 16GB hỗ trợ AMP fp16. Dùng preset `keeplora_bicyc_8gb` (batch 32) cho T4/P100.
  - **Smoke test trước khi chạy thật**: bật `SMOKE_TEST = True` ở cell 5 (Tham số run) — notebook ghim
    batch 16 + 1 epoch/task + cache nhỏ (peak VRAM ~4GB trên T4, không thể OOM). Chạy hết 10 tasks
    ~20–40 phút để xác minh pipeline. Sau đó tắt `SMOKE_TEST` để chạy thật.
  - **Theo dõi VRAM ngay trong log**: sau mỗi task, `run.log` in dòng `GPU alloc=...GiB peak=...GiB`.
    Nếu peak vượt ~85% VRAM card, giảm `BATCH_SIZE` (32 → 16 → 8) ở cell 5.
  - **Checkpoint every epoch**: `checkpoint_every_epochs=1` (mặc định trong notebook) ghi ~350MB mỗi
    epoch; nếu dùng Drive, giảm xuống 0 hoặc tăng khoảng cách để tránh rate-limit.
  - **Resume**: Chạy lại cell 8 (HUẤN LUYỆN) sau khi session reconnect. Nếu thay đổi `EPOCHS_PER_TASK`
    giữa 2 lần, guard tự động báo lỗi thay vì âm thầm huấn luyện sai.
