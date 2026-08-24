# Kiến trúc Hệ thống & Sơ đồ Luồng Hoạt động (Architecture)

Tài liệu này mô tả chi tiết kiến trúc phần mềm, dòng chảy dữ liệu (dataflow) và cơ chế cô lập gradient (gradient barrier) cho toàn bộ hệ thống thí nghiệm **BiCyc Multi-Adapter**.

---

## 1. Sơ đồ Luồng Dữ liệu Tổng thể (End-to-End Dataflow)

```mermaid
graph TD
    subgraph "1. Data Streaming (CIFAR-100 Task Stream)"
        X["Batch Ảnh Hiện Tại x_t"]
        Y["Nhãn Lớp y_t"]
    end

    subgraph "2. Backbone & Multi-Adapter Extraction"
        X --> ViT["Frozen ViT-Base Backbone (timm)"]
        ViT --> Blocks["Transformer Blocks (qkv, proj, fc1, fc2)"]
        Blocks --> Router["PFD Dynamic Router: Φ_{cos}(q, D_j)"]
        Router --> LoRA["RoutedKeepLoRA: ΔW_j = (α/r) A_j (B_j - B_j⁰)"]
        LoRA --> Znew["Current Features z_new ∈ R^768"]
        
        X --> Teacher["Frozen Teacher Snapshot f_{t-1}(x)"]
        Teacher --> Zold["Old Features z_old ∈ R^768 (no_grad)"]
    end

    subgraph "3. Novel Alignment & Gating"
        Znew & Zold --> KL["Per-channel Gaussian-KL: δ_{t,i}"]
        KL --> Gate["Vector Adaptive Gate: λ_{t,i} ∈ [λ_min, λ_max]^d"]
        
        Znew --> D["Distiller Map D: z_new ➔ z_old"]
        Zold --> A["Adapter Map A: z_old ➔ z_new"]
        
        D & Zold & Gate --> Ldistill["Gated Distillation Loss: L_distill"]
        A & D & Znew & Zold --> Lbicyc["Alignment Loss: L_BiCyc + L_iso"]
    end

    subgraph "4. Classification & Inference"
        Znew --> LinearHead["Linear Head ➔ CE Loss (Train Task t)"]
        Znew --> Bayes["Gaussian-Bayes Classifier: S(x, c)"]
        A -.->|"Transport: μ'=Aμ, Σ'=AΣA^T"| Bayes
    end
```

---

## 2. Ranh giới Gradient & Cơ chế 2 Optimizer

Hệ thống bắt buộc tuân thủ nguyên tắc **Cách ly Gradient Tuyệt đối** để vừa bảo tồn tri thức cũ vừa không làm suy giảm khả năng học mới (*Plasticity*):

```mermaid
sequenceDiagram
    autonumber
    participant D as DataLoader (x_t, y_t)
    participant M as Model (ViT + B_t + Head)
    participant T as Teacher Snapshot (f_{t-1})
    participant B as BiCyc Maps (A & D)
    participant O1 as Optimizer 1 (Model)
    participant O2 as Optimizer 2 (Alignment)

    Note over M,B: === BƯỚC 1: MODEL STEP (Tối ưu B_t và Head) ===
    D->>M: Forward x_t ➔ z_new, logits
    D->>T: Forward x_t ➔ z_old (no_grad)
    M->>B: Forward D(z_new) (đóng băng A, D)
    Note over M: Loss_model = CE(logits, y_t) + λ_bi * L_distill(D(z_new), z_old, λ_vec)
    O1->>M: Backprop Loss_model ➔ Cập nhật B_t và Head
    Note over B: Không nhận gradient từ Bước 1!

    Note over M,B: === BƯỚC 2: ALIGNMENT STEP (Tối ưu A và D) ===
    Note over M,T: Detach z_new (sg) và z_old (sg)
    M->>B: Forward A(sg(z_old)), D(sg(z_new)), Cycle, Iso
    Note over B: Loss_maps = λ_bi * L_bi + λ_cyc * L_cyc + λ_iso * L_iso
    O2->>B: Backprop Loss_maps ➔ Chỉ cập nhật trọng số A và D
    Note over M: Không có gradient nào truyền ngược về B_t hay Backbone!
```

---

## 3. Vòng đời Huấn luyện một Task CIL (`DirectionOneExperiment`)

Mỗi task $t \in \{0, \dots, T-1\}$ trải qua 6 giai đoạn có kiểm soát chặt chẽ:

1. **`expand_head`**:
   Mở rộng thêm số hàng trong `nn.Linear` classifier tương ứng với các class mới của task $t$. Các hàng của class cũ được giữ nguyên trọng số.
2. **`begin_task`**:
   * Chạy 1 lượt phân loại ban đầu trên toàn bộ stream dữ liệu của task $t$ để tích lũy ma trận gradient $G_t$.
   * Chiếu trực giao gradient $\hat{G}_t = (I - Q_{t-1}Q_{t-1}^\top) G_t$.
   * Phân tích SVD để khởi tạo $(A_t, B_t^{(0)})$, đóng băng $A_t$ và kích hoạt $B_t$.
   * Gắn forward hooks thu thập kích hoạt đầu vào của 48 target linears (cache trên RAM CPU).
3. **`train_epochs`**:
   * Chạy vòng lặp huấn luyện theo cơ chế 2-Optimizer (`KeepLoRATrainer.train_batch`).
   * Cập nhật vector kỳ vọng trực tuyến $\mathcal{D}_t^l = \mathbb{E}[W^l h^l(x)]$ cho PFD router sau mỗi batch.
4. **`Consolidation`**:
   * Chạy 1 lượt forward qua train set để trích xuất đặc trưng $z_{new}$.
   * Nếu $t > 0$: Vận chuyển thống kê của toàn bộ class cũ qua mạng $A$:
     $$\mu'_c = \mu_c W_A^\top + b_A, \qquad \Sigma'_c = W_A \Sigma_c W_A^\top$$
   * Fit kỳ vọng $\mu_c$ và hiệp phương sai $\Sigma_c$ cho các class mới của task $t$.
5. **`end_task`**:
   * Trích xuất các hướng kích hoạt dư từ cache, phân tích SVD để cập nhật Feature Memory $M_t = \operatorname{orth}([M_{t-1}, U_{t, 1:m}])$.
   * Đóng băng hoàn toàn ma trận $B_t$.
6. **`snapshot` & `_evaluate_upto`**:
   * Tạo bản sao bất biến (deep-copy snapshot) $f_t$ để làm Teacher cho task $t+1$.
   * Đánh giá ma trận độ chính xác trên toàn bộ test set của các task đã học từ $0 \dots t$.
   * Lưu checkpoint an toàn (rolling atomic save).

