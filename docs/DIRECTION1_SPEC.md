# Đặc tả Toán học & Nguyên lý Hoạt động Hướng 1 (`keeplora_bicyc`)

## 1. Tổng quan & Ràng buộc Bài toán

Hướng nghiên cứu 1 giải quyết bài toán **Học tăng cường theo lớp không lưu mẫu (Exemplar-Free Class-Incremental Learning - EFCIL)** trên mô hình thị giác nền tảng **Vision Transformer (ViT)** đóng băng:
* **Mục tiêu**: Huấn luyện mô hình nhận diện tuần tự $T$ task mới (ví dụ 10 tasks, mỗi task 10 classes trên CIFAR-100).
* **Ràng buộc nghiêm ngặt**: **Tuyệt đối không lưu lại bất kỳ ảnh mẫu cũ nào ($0$ exemplars)** do yêu cầu bảo mật quyền riêng tư và giới hạn bộ nhớ.
* **Bản chất khoa học**: Đây là một phương pháp **Hybrid sáng tạo** kết hợp sức mạnh bảo vệ tham số của **KeepLoRA (ICLR 2026)**, căn chỉnh biểu diễn hai chiều của **BiCyc (ICLR 2026)**, định tuyến adapter động của **Presentative Feature Distributions - PFD (ICML 2025)** cùng **đóng góp đề xuất mới** của nhóm: **Vector Channel-wise Adaptive Distribution Gate** và **Isometric Transport Regularizer**.

---

## 2. Hệ thống Công thức Toán học Chi tiết

```text
               ┌────────────────────────────────────────────────────────┐
               │              Input Batch x_t (Current Task)            │
               └───────────────┬────────────────────────┬───────────────┘
                               │                        │
                               ▼                        ▼
               ┌────────────────────────┐      ┌────────────────────────┐
               │ Current Model f_t(x)   │      │ Frozen Teacher f_{t-1} │
               │ (ViT + Routed KeepLoRA)│      │ (Immutable Snapshot)   │
               └───────────────┬────────┘      └────────┬───────────────┘
                               │                        │
                               ▼ z_new                  ▼ z_old
               ┌────────────────────────────────────────────────────────┐
               │    1. Channel-wise Gaussian-KL Divergence δ_{t,i}      │
               │    2. Vector Adaptive Gate λ_{t,i} ∈ [λ_min, λ_max]^d   │
               └───────────────────┬────────────────────────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                             ▼
        ┌───────────────────────┐   ┌───────────────────────────┐
        │  Model Step (Opt 1)   │   │   Alignment Step (Opt 2)  │
        │  • CE Loss            │   │   • L_{bi} (A & D maps)   │
        │  • Gated Distillation │   │   • L_{cyc} (Cycle Stab.) │
        │  • Train: B_t & Head  │   │   • L_{iso} (Norm & Dir.) │
        │  • Freeze: A_t, Maps  │   │   • Detached Features (sg)│
        └───────────────────────┘   └───────────────────────────┘
```

### 2.1. Phân rã Trọng số & Chiếu Residual Gradient (KeepLoRA)

Với một lớp tuyến tính gốc dùng quy ước $y = x W$, trong đó $W \in \mathbb{R}^{d_{in} \times d_{out}}$:

1. **Trích xuất không gian trọng số cơ sở ($W_p$)**:
   Thực hiện SVD trên trọng số ban đầu $W$:
   $$W = U \Sigma V^\top, \qquad W_p = U_{:, 1:p}, \qquad \frac{\|\Sigma_{1:p}\|_F^2}{\|\Sigma\|_F^2} \ge \epsilon_w$$
   *Trong đó $\epsilon_w \in (0, 1]$ là ngưỡng bảo toàn năng lượng trọng số (`weight_energy`).*

2. **Hợp nhất không gian bảo vệ trực chuẩn ($Q_{t-1}$)**:
   Gộp $W_p$ với cơ sở đặc trưng lịch sử tích lũy $M_{t-1} \in \mathbb{R}^{d_{in} \times m}$ và trực chuẩn hóa bằng phân rã QR:
   $$Q_{t-1} = \operatorname{orth}([W_p, M_{t-1}]) = \operatorname{QR}([W_p, M_{t-1}]).Q$$

3. **Chiếu Residual Gradient và Khởi tạo LoRA**:
   Trước khi huấn luyện task $t$, thu thập ma trận gradient trung bình $G_t = \frac{\partial L_{CE}}{\partial W}$ từ một lượt quét phân loại ban đầu. Chiếu trực giao $G_t$ ra khỏi không gian bảo vệ $Q_{t-1}$:
   $$\hat{G}_t = (I - Q_{t-1} Q_{t-1}^\top) G_t$$
   Phân tích SVD trên gradient dư:
   $$\hat{G}_t = \bar{U} \bar{\Sigma} \bar{V}^\top$$
   Khởi tạo các ma trận nhân tử rank-$r$:
   $$A_t = \bar{U}_{:, 1:r} \in \mathbb{R}^{d_{in} \times r}, \qquad B_t^{(0)} = \bar{\Sigma}_{1:r, 1:r} \bar{V}_{:, 1:r}^\top \in \mathbb{R}^{r \times d_{out}}$$
   * **Đóng băng vĩnh viễn $A_t$ (`requires_grad = False`)**, chỉ tối ưu hóa $B_t$.
   * **Function-Preserving Parameterization**: Đầu ra thích ứng tại thời điểm $t$ có dạng:
     $$\Delta W_t = \frac{\alpha}{r} A_t (B_t - B_t^{(0)})$$
     Tại $B_t = B_t^{(0)}$, $\Delta W_t = 0 \implies$ hàm đầu ra bảo toàn $100\%$ biểu diễn trước khi train.

4. **Cập nhật Bộ nhớ Không gian Đặc trưng ($M_t$)**:
   Cuối task $t$, thu thập các vector kích hoạt đầu vào $X_t \in \mathbb{R}^{d_{in} \times N}$ qua forward hooks:
   $$\hat{X}_t = (I - Q_{t-1} Q_{t-1}^\top) X_t = U_t \Sigma_t V_t^\top$$
   Chọn $m$ hướng chủ đạo thỏa mãn $\frac{\|\Sigma_{t, 1:m}\|_F^2}{\|\Sigma_t\|_F^2} \ge \epsilon_f$, sau đó cập nhật:
   $$M_t = \operatorname{orth}([M_{t-1}, U_{t, :, 1:m}])$$

---

### 2.2. Định tuyến Động Đa Adapter (Presentative Feature Distributions - PFD)

Thay vì merge $\Delta W_t$ vào $W$ làm mất tính độc lập, mô hình duy trì một ngân hàng gồm các adapter riêng biệt $\{ (A_j, B_j) \}_{j=0}^t$:

1. **Thống kê Phân phối Trực tuyến ($\mathcal{D}_k^l$)**:
   Với layer $l$ và task $k$, thống kê trung bình đặc trưng đầu ra được cập nhật online không lưu sample:
   $$\mathcal{D}_k^l = \mathbb{E}_{(x,y) \sim p_k}[W^l h^l(x)] \in \mathbb{R}^{d_{out}}$$

2. **Định tuyến Cosine / Similarity**:
   Với đặc trưng đầu vào $q = W^l h^l(x)$, tính điểm tương đồng với prototype của các task $j \in \{0, \dots, t\}$:
   $$\Phi(q, \mathcal{D}_j^l) = \begin{cases}
   \frac{q^\top \mathcal{D}_j^l}{\|q\|_2 \|\mathcal{D}_j^l\|_2} & \text{(Cosine, chuẩn hóa biên độ)} \\
   -\|q - \mathcal{D}_j^l\|_2 & \text{(L2 Distance)} \\
   \frac{q^\top \mathcal{D}_j^l}{\sqrt{d_{out}}} & \text{(Scaled Dot-Product)}
   \end{cases}$$

3. **Tổng hợp Đầu ra Top-K**:
   $$w_j = \frac{\exp(\Phi(q, \mathcal{D}_j^l) / T)}{\sum_{v \in \operatorname{TopK}} \exp(\Phi(q, \mathcal{D}_v^l) / T)}$$
   $$y = W^l h^l(x) + \sum_{j \in \operatorname{TopK}} w_j \cdot \Delta W_j^l h^l(x)$$
   *(Cấu hình `TopK = 1` và $T = 0.1$ triệt tiêu hoàn toàn hiện tượng rò rỉ adapter giữa các task).*

---

### 2.3. Căn chỉnh BiCyc Hai Chiều & Ràng buộc Isometric

Mô hình học hai mạng MLP đối ngẫu: **Adapter** $A: z_{old} \to z_{new}$ và **Distiller** $D: z_{new} \to z_{old}$, trong đó $z_{old} = f_{t-1}(x)$ và $z_{new} = f_t(x)$:

1. **Bidirectional Alignment Loss**:
   $$L_{bi} = \|D(z_{new}) - \operatorname{sg}(z_{old})\|_2^2 + \|A(\operatorname{sg}(z_{old})) - \operatorname{sg}(z_{new})\|_2^2$$

2. **Cycle Consistency Loss**:
   $$L_{cyc} = \|A(D(\operatorname{sg}(z_{new}))) - \operatorname{sg}(z_{new})\|_2^2 + \|D(A(\operatorname{sg}(z_{old}))) - \operatorname{sg}(z_{old})\|_2^2$$

3. **Isometric & Direction Regularization ($L_{iso}$ — Đóng góp mới)**:
   Để chống lại sai số tích lũy làm co rút hoặc bùng nổ phương sai khi transport qua chuỗi $T$ tasks:
   $$L_{iso} = \left( \frac{\|A(z_{old})\|_2}{\|z_{old}\|_2 + \epsilon} - 1 \right)^2 + \left( 1 - \frac{z_{old}^\top A(z_{old})}{\|z_{old}\|_2 \|A(z_{old})\|_2 + \epsilon} \right)$$

4. **Tổng Hàm Mất Mát Alignment**:
   $$L_{BiCyc} = \lambda_{bi} L_{bi} + \lambda_{cyc} L_{cyc} + \lambda_{iso} L_{iso}$$

---

### 2.4. Vector Channel-wise Adaptive Distribution Gate (Đóng góp mới)

Ước lượng phân phối Gaussian đường chéo $q_{old} = \mathcal{N}(\mu_o, \operatorname{diag}(\sigma_o^2))$ và $q_{new} = \mathcal{N}(\mu_n, \operatorname{diag}(\sigma_n^2))$ trên chính batch hiện tại.

1. **Khoảng cách Đối xứng Gaussian-KL theo từng chiều ($i \in \{1, \dots, d\}$)**:
   $$\delta_{t, i} = \frac{1}{2}\left[ \frac{\sigma_{o,i}^2 + (\mu_{o,i} - \mu_{n,i})^2}{\sigma_{n,i}^2} + \frac{\sigma_{n,i}^2 + (\mu_{n,i} - \mu_{o,i})^2}{\sigma_{o,i}^2} - 2 \right]$$

2. **Vector Trọng số Thích ứng $\vec{\lambda}_t \in [\lambda_{min}, \lambda_{max}]^d$**:
   $$\vec{\lambda}_{t, i} = \lambda_{min} + (\lambda_{max} - \lambda_{min})\left(1 - \exp\left(-\frac{\delta_{t, i}}{\tau}\right)\right)$$

3. **Gated Distillation Loss cho Model Step**:
   $$L_{distill} = \frac{1}{B} \sum_{b=1}^B \sum_{i=1}^d \vec{\lambda}_{t, i} \cdot \left(D(z_{new})_i - z_{old, i}\right)^2$$
   *Ý nghĩa*: Chiều đặc trưng nào ít biến động giữa 2 task sẽ được bảo tồn triệt để ($\vec{\lambda}_{t,i} \approx \lambda_{max}$), chiều nào có độ lệch lớn sẽ giảm ràng buộc để tối đa hóa khả năng tiếp thu tri thức mới (*Plasticity*).

---

### 2.5. Cơ chế Hai Optimizer Độc lập & Ranh giới Gradient

Nhằm bảo đảm tính toàn vẹn của không gian đặc trưng mới, hai optimizer được thực thi tách biệt tuyệt đối trong mỗi batch:

$$\begin{aligned}
\text{\bf Bước 1 (Model Step):} \quad & L_{model} = L_{CE}(logits, y) + \lambda_{bi} \cdot L_{distill} \\
& \implies \text{Cập nhật: } B_t \text{ và Linear Head. Đóng băng: } A_t, A, D. \\
\text{\bf Bước 2 (Alignment Step):} \quad & L_{maps} = L_{BiCyc}(\operatorname{sg}(z_{old}), \operatorname{sg}(z_{new})) \\
& \implies \text{Cập nhật: } A \text{ và } D. \text{ Tuyệt đối không truyền gradient về } B_t \text{ hay Backbone.}
\end{aligned}$$

---

### 2.6. Gaussian Transport cho Bộ phân loại CIL (Exemplar-Free)

Mỗi class $c$ được đặc trưng hóa bởi cặp thống kê kỳ vọng và hiệp phương sai $(\mu_c, \Sigma_c)$ thay vì lưu sample thô:

1. **Vận chuyển Thống kê Cũ qua Ánh xạ Affine $A$**:
   Khi mô hình chuyển từ $f_{t-1}$ sang $f_t$, với mạng $A: z \mapsto z W_A^\top + b_A$:
   $$\mu'_c = \mu_c W_A^\top + b_A, \qquad \Sigma'_c = W_A \Sigma_c W_A^\top$$
   *(Chế độ `diagonal`: $\operatorname{Var}'_{c, i} = \sum_j W_{A, ij}^2 \operatorname{Var}_{c, j}$).*

2. **Fit Thống kê Mới của Task Hiện tại**:
   $$\mu_c = \frac{1}{N_c} \sum_{x \in \mathcal{C}_c} z_{new}(x), \qquad \text{Scatter}_c = \sum_{x \in \mathcal{C}_c} (z_{new}(x) - \mu_c)(z_{new}(x) - \mu_c)^\top$$

3. **Chấm điểm Gaussian-Bayes Inference**:
   $$S(x, c) = -\frac{1}{2} \left[ d \ln(2\pi) + \ln|\Sigma_c + \gamma I| + (z - \mu_c)^\top (\Sigma_c + \gamma I)^{-1} (z - \mu_c) \right]$$
   $$\hat{y} = \arg\max_{c \in \text{Seen Classes}} S(x, c)$$

---

## 3. Bảng Ánh xạ: Công thức Toán $\longleftrightarrow$ Mã Nguồn

| Thành phần Thuật toán | Công thức Toán học | Tệp Mã Nguồn | Hàm / Lớp Phụ trách |
| :--- | :---: | :--- | :--- |
| **KeepLoRA SVD Init** | $\hat{G}_t = (I - QQ^\top)G_t = U\Sigma V^\top$ | [`models/adapters/keeplora.py`](file:///d:/MyProject/BiCyc_MultiAdapter/src/bicyc_multiadapter/models/adapters/keeplora.py) | `initialize_lora_from_gradient`, `residual_gradient` |
| **Feature Memory Update** | $M_t = \operatorname{orth}([M_{t-1}, U_{t, 1:m}])$ | [`models/adapters/keeplora.py`](file:///d:/MyProject/BiCyc_MultiAdapter/src/bicyc_multiadapter/models/adapters/keeplora.py) | `update_feature_subspace` |
| **PFD Router Statistics** | $\mathcal{D}_k^l = \mathbb{E}[W^l h^l(x)]$ | [`models/adapters/routing.py`](file:///d:/MyProject/BiCyc_MultiAdapter/src/bicyc_multiadapter/models/adapters/routing.py) | `PresentativeFeatureRouter.update_distribution` |
| **Cosine Dynamic Routing** | $\Phi_{cos}(q, \mathcal{D}_j) = \frac{q^\top \mathcal{D}_j}{\|q\|_2 \|\mathcal{D}_j\|_2}$ | [`models/adapters/routing.py`](file:///d:/MyProject/BiCyc_MultiAdapter/src/bicyc_multiadapter/models/adapters/routing.py) | `PresentativeFeatureRouter.routing_weights` |
| **Channel-wise Gate** | $\vec{\lambda}_{t, i} = \lambda_{min} + \Delta\lambda(1 - e^{-\delta_{t,i}/\tau})$ | [`models/alignment/distribution.py`](file:///d:/MyProject/BiCyc_MultiAdapter/src/bicyc_multiadapter/models/alignment/distribution.py) | `adaptive_alignment_weight`, `channelwise_symmetric_gaussian_kl` |
| **BiCyc Alignment & Iso** | $L_{BiCyc} = \lambda_{bi}L_{bi} + \lambda_{cyc}L_{cyc} + \lambda_{iso}L_{iso}$ | [`models/alignment/bicyc.py`](file:///d:/MyProject/BiCyc_MultiAdapter/src/bicyc_multiadapter/models/alignment/bicyc.py) | `bicyc_loss`, `isometric_regularization_loss` |
| **2-Optimizer Trainer** | Step 1 ($L_{model}$) & Step 2 ($L_{maps}$) | [`engine/keeplora_trainer.py`](file:///d:/MyProject/BiCyc_MultiAdapter/src/bicyc_multiadapter/engine/keeplora_trainer.py) | `KeepLoRATrainer.train_batch` |
| **Gaussian Transport** | $\mu' = A\mu, \Sigma' = A\Sigma A^\top$ | [`models/classifier.py`](file:///d:/MyProject/BiCyc_MultiAdapter/src/bicyc_multiadapter/models/classifier.py) | `GaussianCILClassifier.transport`, `fit_task`, `scores` |
| **Vòng đời Task CIL** | Lifecycle 10 Tasks Pipeline | [`engine/task_loop.py`](file:///d:/MyProject/BiCyc_MultiAdapter/src/bicyc_multiadapter/engine/task_loop.py) | `DirectionOneExperiment.run`, `_evaluate_upto` |

---

## 4. Thiết kế Thực nghiệm & Ablation Study

Nhóm duy trì 5 cấu hình thí nghiệm để báo cáo đóng góp khoa học một cách minh bạch:
1. **`keeplora_original`**: KeepLoRA nguyên gốc (merge adapter sau task, không distillation).
2. **`KeepLoRA + Distiller D (backward-only)`**: Distillation 1 chiều thông thường ($\lambda_t = 1$).
3. **`KeepLoRA + Full BiCyc (Fixed)`**: Căn chỉnh 2 chiều cố định $\lambda_t = 1$.
4. **`KeepLoRA + BiCyc + PFD Routing (Scalar Gate)`**: Bản kết hợp với scalar adaptive gate.
5. **`keeplora_bicyc` (Proposed Full)**: Đầy đủ Multi-Adapter + BiCyc + Isometric Regularizer + Channel-wise Vector Gate + Cosine Router.

