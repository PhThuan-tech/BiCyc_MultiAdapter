# Đặc tả hướng 1 — KeepLoRA + BiCyc + adaptive distribution gate

## Phạm vi và nguồn

Phần này ghép **KeepLoRA**, **BiCyc** và **Presentative Feature Distributions (PFD)** cho EFCIL. Đây không phải tái hiện nguyên xi một paper: KeepLoRA gốc nhắm continual adaptation cho mô hình pretrained, BiCyc gốc dùng backbone CIL + Gaussian Bayes, còn PFD gốc thực hiện routing LoRA cho LLM. Phần `adaptive distribution gate` là giả thuyết nghiên cứu của nhóm và phải được báo cáo là module mới.

- KeepLoRA: residual gradient, SVD initialization, freeze-A/train-B; bản gốc merge ở cuối task.
- BiCyc: adapter \(A:z_{old}\rightarrow z_{new}\), distiller \(D:z_{new}\rightarrow z_{old}\), stop-gradient, cycle consistency, và transport Gaussian statistics.
- PFD: đặc trưng hóa mỗi adapter bằng trung bình feature của backbone frozen, sau đó chọn/mix adapter bằng similarity không có selector trainable.
- Distribution gate: ablation proposal tùy chọn, quyết định cường độ distillation từ độ gần phân phối của feature cũ/mới trên **chính batch task hiện tại**; không cần giữ raw exemplar cũ.

## Công thức bắt buộc

Với một linear layer dùng quy ước \(y=xW\), \(W\in\mathbb{R}^{d_{in}\times d_{out}}\), KeepLoRA:

\[
W=U\Sigma V^\top,\qquad W_p=U_{:,1:p},\qquad
\frac{\|\Sigma_{1:p}\|_F^2}{\|\Sigma\|_F^2}\geq\epsilon_w.
\]

Gộp \(W_p\) với feature basis đã lưu \(M_{t-1}\), trực chuẩn hóa thành \(Q_{t-1}=\operatorname{orth}([W_p,M_{t-1}])\). Viết QR là dạng ổn định số của phép trừ hai projection trong paper:

\[
\hat G_t=(I-Q_{t-1}Q_{t-1}^\top)G_t,
\qquad \hat G_t=\bar U\bar\Sigma\bar V^\top.
\]

\[
A_t=\bar U_{:,1:r},\qquad B_t=\bar\Sigma_{1:r,1:r}\bar V_{:,1:r}^{\top}.
\]

Để output của layer ban đầu không đổi:

\[
W' = W-\frac{\alpha}{r}A_tB_t,\qquad
y=x\left(W'+\frac{\alpha}{r}A_tB_t\right).
\]

Freeze \(A_t\), chỉ tối ưu \(B_t\). Cuối task bản gốc merge \(W\leftarrow W'+\frac{\alpha}{r}A_tB_t\). Code dùng tham số hóa tương đương \(\frac{\alpha}{r}A_t(B_t-B_t^{(0)})\): output ban đầu không đổi, đồng thời factor có thể được giữ trong bank nhiều adapter. Với input activations \(X_t\in\mathbb{R}^{d_{in}\times n}\), cập nhật compact feature memory:

\[
\hat X_t=(I-Q_{t-1}Q_{t-1}^\top)X_t,
\quad \hat X_t=U_t\Sigma_tV_t^\top,
\quad M_t=\operatorname{orth}([M_{t-1},U_{t,:,1:m}]).
\]

Chọn \(m\) theo energy threshold \(\epsilon_f\). Chỉ checkpoint basis, không checkpoint raw activations/dataset.

### Presentative Feature Distribution (PFD) cho multi-adapter

Với layer frozen \(W^l\), input hidden state \(h^l(x)\), statistic của task \(k\) là Eq. (3) của bài PFD:

\[
\mathcal D_k^l=\mathbb E_{(x,y)\sim p_k}[W^l h^l(x)].
\]

Chỉ cập nhật online mean, không lưu sample. Với \(q=W^lh^l(x)\), điểm routing là \(\Phi_{L2}(q,\mathcal D_k^l)=-\|q-\mathcal D_k^l\|_2\) hoặc \(\Phi_{dot}=q^\top\mathcal D_k^l/\sqrt{d_{out}}\). Output của layer được mix theo Eq. (6):

\[
y=W^lh^l(x)+\sum_{j\in\operatorname{TopK}}
\frac{\exp(\Phi(q,\mathcal D_j^l)/T)}
{\sum_{v\in\operatorname{TopK}}\exp(\Phi(q,\mathcal D_v^l)/T)}
\Delta W_j^lh^l(x).
\]

Khác biệt có chủ đích so với KeepLoRA gốc: **không merge** ở cuối task mà freeze factor task \(j\) và distribution mean \(\mathcal D_j\) để route lúc train/inference. Đây là đóng góp hybrid, phải có ablation `merge=true` (KeepLoRA nguyên gốc) và `routed_multi_adapter`.

Với same current-task image \(x\):

\[
z_{old}=f_{t-1}(x),\qquad z_{new}=f_t(x).
\]

\[
L_{bi}=\|D(z_{new})-\operatorname{sg}(z_{old})\|_2^2+
\|A(\operatorname{sg}(z_{old}))-\operatorname{sg}(z_{new})\|_2^2.
\]

\[
L_{cyc}=\|A(D(\operatorname{sg}(z_{new})))-\operatorname{sg}(z_{new})\|_2^2+
\|D(A(\operatorname{sg}(z_{old})))-\operatorname{sg}(z_{old})\|_2^2.
\]

\[
L_{BiCyc}=\lambda_{bi}L_{bi}+\lambda_{cyc}L_{cyc}.
\]

### Adaptive distribution gate — đề xuất mới

Fit Gaussian đường chéo \(q_{old}=\mathcal N(\mu_o,\operatorname{diag}(\sigma_o^2))\) và \(q_{new}\) từ paired feature của batch. Với \(\delta_t=\frac12[\mathrm{KL}(q_o\|q_n)+\mathrm{KL}(q_n\|q_o)]\):

\[
\lambda_t=\lambda_{min}+(\lambda_{max}-\lambda_{min})\exp(-\delta_t/\tau).
\]

Loss cho B/encoder và map tách thành hai optimizer:

\[
L_{model}=L_{CE}+\lambda_t\lambda_{bi}\|D(z_{new})-\operatorname{sg}(z_{old})\|_2^2.
\]

\[
L_{maps}=L_{BiCyc}(\operatorname{sg}(z_{old}),\operatorname{sg}(z_{new})).
\]

Không cho \(L_{cyc}\), forward map, hoặc alignment optimizer truyền gradient về \(B_t\). Đây là ràng buộc quan trọng để không kéo feature mới về không gian cũ và làm mất plasticity.

## Mapping công thức → mã nguồn

| Thành phần | File | Trạng thái |
| --- | --- | --- |
| Weight/feature SVD, residual projection, factor init, merge | `models/adapters/keeplora.py` | Đã code |
| PFD mean, L2/dot similarity, softmax/Top-K routing | `models/adapters/routing.py` | Đã code — multi-adapter hybrid |
| A/D, \(L_{bi}\), \(L_{cyc}\), stop-gradient, anti-collapse | `models/alignment/bicyc.py` | Đã code |
| Gaussian diagonal symmetric-KL và \(\lambda_t\) | `models/alignment/distribution.py` | Đã code — proposal |
| Hai bước optimizer, chỉ đường gradient được phép | `engine/keeplora_trainer.py` | Đã code — generic contract |
| Cấu hình và hyperparameters ban đầu | `configs/experiment/keeplora_bicyc.yaml` | Đã cập nhật |

## Việc còn phải nối với backbone/dataset

1. Chọn các `nn.Linear` của ViT/CLIP cần thay bằng `RoutedKeepLoRALinear`; chuyển trọng số PyTorch từ `[d_out,d_in]` sang `[d_in,d_out]` khi khởi tạo. Bản baseline merge dùng `FrozenAResidualLoRA` rồi chuyển ngược khi merge.
2. Trước task \(t\), chạy **một** backward CE-only để lấy \(G_t\) cho mỗi layer đích; dùng `initialize_lora_from_gradient` trước vòng train chính. Không dùng raw old data.
3. Gắn forward hook thu activation input của từng layer đích trong task và gọi `update_feature_subspace` ở cuối task.
4. Snapshot `old_model` ngay trước task mới; feature cũ chỉ sinh từ ảnh của task hiện tại đi qua snapshot này, hợp lệ trong exemplar-free CIL. Update \(\mathcal D_t^l\) online trong task, sau đó freeze task adapter và distribution mean.
5. Nếu dùng CIL classifier Gaussian của BiCyc, lưu \((\mu_c,\Sigma_c)\) thay vì samples; transport class cũ qua \(A\): \(\mu'_c=A\mu_c\), \(\Sigma'_c=A\Sigma_cA^\top\).

## Ablation tối thiểu

1. KeepLoRA only: CE, không BiCyc.
2. KeepLoRA + backward-only \(D\), \(\lambda_t=1\).
3. KeepLoRA + full BiCyc fixed \(\lambda_t=1\).
4. KeepLoRA + BiCyc + PFD routing (L2), bỏ adaptive gate.
5. KeepLoRA + BiCyc + PFD routing + proposed Gaussian-KL gate.

Các run giữ cố định class order, rank, energy threshold, epoch budget và ít nhất 3 seeds.
