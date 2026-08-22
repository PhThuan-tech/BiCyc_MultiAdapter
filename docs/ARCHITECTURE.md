# Kiến trúc source code

```text
image -> Frozen backbone -> adapter -> classifier -> CE loss
                  |              |
                  |              +-> current feature
                  +-> old snapshot feature (no_grad, only incremental tasks)
```

| Nhánh | Mã nguồn chính | Đường gradient bắt buộc |
| --- | --- | --- |
| KeepLoRA + Adaptive BiCyc | `adapters/keeplora.py`, `alignment/bicyc.py`, `engine/keeplora_trainer.py` | Old snapshot không nhận gradient. A/D alignment và LoRA update là hai bước tách biệt. |
| RSIAT + Bi-RAE | `adapters/rsiat.py`, `alignment/bi_rae.py`, `engine/rsiat_trainer.py` | Shared adapter nhận CE + RS/orthogonal + Bi-RAE loss. Bi-RAE không được dùng khi evaluate. |

## Hướng 1: KeepLoRA + BiCyc + PFD routing

KeepLoRA, BiCyc và Presentative Feature Distributions (PFD) không phải cùng một paper; hướng 1 là hybrid đề xuất. KeepLoRA/BiCyc/PFD phải giữ đúng công thức gốc, còn adaptive Gaussian-KL gate được tách riêng để ablation công bằng.

Với một linear layer, dùng quy ước `x @ W`, \(W\in\mathbb{R}^{d_{in}\times d_{out}}\). KeepLoRA tạo principal weight basis \(W_p\) bằng SVD, gộp nó với basis feature cũ \(M_{t-1}\) thành basis trực chuẩn \(Q\), rồi:

\[
\hat G_t = (I - QQ^\top)G_t,\quad
\hat G_t=U\Sigma V^\top,\quad
A_t=U_{:,1:r},\quad B_t=\Sigma_{1:r,1:r}V_{:,1:r}^\top.
\]

Khởi tạo function-preserving dùng \(W'=W-\frac{\alpha}{r}A_tB_t\), freeze \(A_t\), train \(B_t\), và merge \(W' +\frac{\alpha}{r}A_tB_t\) ở cuối task. Xem `adapters/keeplora.py`.

BiCyc dùng \(z_{old}=f_{t-1}(x)\), \(z_{new}=f_t(x)\), adapter \(A:z_{old}\to z_{new}\) và distiller \(D:z_{new}\to z_{old}\):

\[
L_{bi}=\|D(z_{new})-z_{old}\|_2^2+
\|A(z_{old})-\operatorname{sg}(z_{new})\|_2^2
\]

\[
L_{cyc}=\|A(D(\operatorname{sg}(z_{new})))-\operatorname{sg}(z_{new})\|_2^2+
\|D(A(z_{old}))-\operatorname{sg}(z_{old})\|_2^2.
\]

\[
L_{BiCyc}=\lambda_{bi}L_{bi}+\lambda_{cyc}L_{cyc}.
\]

Adaptive gate là đóng góp được đề xuất, không phải công thức của hai bài: fit Gaussian đường chéo trên paired features hiện tại, tính symmetric KL \(\delta_t\), rồi \(\lambda_t=\lambda_{min}+(\lambda_{max}-\lambda_{min})e^{-\delta_t/\tau}\). Như vậy task gần nhau bị căn chỉnh mạnh, task xa nhau ưu tiên plasticity. Xem `alignment/distribution.py`.

PFD cung cấp phần multi-adapter: mỗi task giữ một mean statistic \(\mathcal D_k^l=\mathbb E[W^lh^l(x)]\). Feature hiện tại được so với các mean bằng negative-L2 hoặc dot-product, sau đó softmax/Top-K để mix các LoRA block. Vì KeepLoRA gốc merge adapter sau task còn PFD yêu cầu giữ adapter, `RoutedKeepLoRALinear` là biến thể nghiên cứu và cần được so sánh trực tiếp với KeepLoRA merge baseline.

## Thứ tự hiện thực khuyến nghị

1. Hiện thực `data/` và backbone frozen, sau đó baseline classifier CIL tối giản.
2. Hiện thực KeepLoRA (SVD/projection) cùng test gradient isolation trước BiCyc.
3. Hiện thực BiCyc fixed, sau đó adaptive coefficient và ablation.
4. Hiện thực shared adapter + RS warm-up, rồi mới thêm Bi-RAE.
5. Đưa toàn bộ metric vào `evaluation/`, chạy ba seed cho mỗi cấu hình.

Mỗi nhánh có trainer độc lập bởi vì nghiệm thu gradient và lifecycle module khác nhau; chỉ dùng chung protocol dữ liệu, backbone, metric và tiện ích tái lập.
