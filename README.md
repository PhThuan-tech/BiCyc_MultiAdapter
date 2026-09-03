# BiCyc Multi-Adapter: Exemplar-Free Class-Incremental Learning (EFCIL)

Hệ thống mã nguồn và pipeline thực nghiệm cho bài toán **Học tăng cường theo lớp không lưu mẫu (Exemplar-Free Class-Incremental Learning - EFCIL)** trên mô hình thị giác nền tảng **Vision Transformer (ViT)** đóng băng.

Tài liệu này tập trung toàn bộ vào **Hướng nghiên cứu số 1 (`keeplora_bicyc`)**: Kiến trúc lai kết hợp **KeepLoRA** đa adapter, định tuyến phân phối đặc trưng đại diện (**PFD Router**), căn chỉnh hai chiều (**BiCyc**), cùng các đóng góp lý thuyết mới (**Vector Channel-wise Adaptive Gate** và **Isometric Regularizer**).

---

## 1. Tóm tắt Hướng Nghiên cứu Số 1 (`keeplora_bicyc`)

### 1.1. Bản chất Bài toán
* **Mục tiêu**: Huấn luyện mô hình ViT nhận diện tuần tự $T$ tác vụ phân loại lớp mới (chuẩn thực nghiệm: 10 tasks CIFAR-100, mỗi task gồm 10 lớp phân biệt).
* **Ràng buộc nghiêm ngặt**: **Tuyệt đối không lưu trữ bất kỳ ảnh mẫu cũ nào ($0$ exemplars)** do yêu cầu bảo mật quyền riêng tư và giới hạn bộ nhớ.
* **Thách thức cốt lõi**: Giải quyết nghịch lý giữa tính bảo tồn tri thức cũ (*Stability*) và tính mềm dẻo tiếp thu khái niệm mới (*Plasticity*), loại bỏ hiện tượng *quên thảm họa (catastrophic forgetting)*.

### 1.2. Cơ chế Hoạt động & Đóng góp Khoa học Đề xuất
Phương pháp đề xuất giải quyết bài toán qua 5 trụ cột kỹ thuật:

```text
                  Input Batch x_t (Current Task Stream)
                     │                            │
                     ▼                            ▼
        ┌─────────────────────────┐  (no-grad) ┌─────────────────────────┐
        │  Current Model f_t(x)   │            │ Teacher Snapshot f_{t-1} │
        │  (ViT + RoutedKeepLoRA) │            │ (Immutable Snapshot)    │
        └────────────┬────────────┘            └────────────┬────────────┘
                     │ z_new                                │ z_old
                     ▼                                      ▼
        ┌────────────────────────────────────────────────────────────────┐
        │ 1. Channel-wise Gaussian-KL Divergence δ_{t, i}                │
        │ 2. Vector Adaptive Distribution Gate λ_{t, i} ∈ [λ_min, λ_max] │
        └──────────────────────────────┬─────────────────────────────────┘
                                       │
                ┌──────────────────────┴──────────────────────┐
                ▼                                             ▼
    ┌───────────────────────────────┐           ┌───────────────────────────────┐
    │     BƯỚC 1: MODEL STEP        │           │   BƯỚC 2: ALIGNMENT STEP      │
    │  (Optimizer 1 - Train B, Head)│           │    (Optimizer 2 - Train A, D) │
    │  • L_CE(logits, y_t)          │           │  • L_bi(A, D on sg(z))        │
    │  • L_distill(D(z_new), z_old, │           │  • L_cyc(Cycle consistency)   │
    │              vector λ_{t, i}) │           │  • L_iso (Norm & Direction)   │
    │  • Freeze: A_t, Maps A, D     │           │  • Detached Inputs (sg)       │
    └───────────────────────────────┘           └───────────────────────────────┘
```

1. **KeepLoRA Residual Gradient SVD**:
   * Tích hợp các adapter rank thấp vào các tầng tuyến tính (`qkv`, `proj`, `fc1`, `fc2`) của ViT đóng băng.
   * Chiếu gradient dư vào không gian trực giao với không gian trọng số cơ sở $W_p$ và không gian kích hoạt lịch sử $M_{t-1}$: $\hat{G}_t = (I - Q_{t-1}Q_{t-1}^\top) G_t$.
   * Khởi tạo $A_t$ và $B_t^{(0)}$ qua SVD; đóng băng vĩnh viễn $A_t$, chỉ huấn luyện $B_t$ với công thức bảo toàn hàm $\Delta W_t = \frac{\alpha}{r} A_t (B_t - B_t^{(0)})$.
2. **Ngân hàng Đa Adapter & Định tuyến Động PFD (Cosine Router)**:
   * Không gộp (*merge*) adapter vào backbone để tránh suy thoái biểu diễn; duy trì một ngân hàng gồm $t+1$ adapter riêng biệt.
   * Tích lũy kỳ vọng trực tuyến $\mathcal{D}_k^l = \mathbb{E}[W^l h^l(x)]$ không lưu mẫu; khi suy diễn, sử dụng độ tương đồng Cosine Top-K để tự động định tuyến đến đúng adapter chuyên biệt.
3. **Căn chỉnh Hai chiều BiCyc & Ràng buộc Đẳng cự ($L_{iso}$ - Đề xuất mới)**:
   * Huấn luyện hai mạng affine đối ngẫu: Adapter $A: z_{old} \to z_{new}$ và Distiller $D: z_{new} \to z_{old}$ kết hợp mất mát chu trình đối xứng $\mathcal{L}_{cyc}$.
   * **Đóng góp mới**: Bổ sung hàm phạt đẳng cự $\mathcal{L}_{iso}$ kiểm soát biến dạng độ dài chuẩn vector và chống lệch góc hướng xoay, ngăn chặn hiện tượng bùng nổ hoặc co rút phương sai khi vận chuyển phân phối qua nhiều tác vụ liên tiếp.
4. **Cổng Phân phối Thích ứng Theo Kênh (Channel-wise Gate - Đề xuất mới)**:
   * Đo lường phân kỳ đối xứng Gaussian-KL theo từng chiều đặc trưng $\delta_{t, i}$ trên batch hiện tại.
   * Sinh vector trọng số $\vec{\lambda}_{t, i} \in [\lambda_{min}, \lambda_{max}]^d$. Các kênh có biến động nhỏ (đặc trưng ngữ nghĩa chung) được bảo tồn triệt để, các kênh có độ lệch lớn được giảm lực cản chưng cất để mô hình tự do tiếp nhận khái niệm mới.
5. **Cơ chế Cô lập Gradient Hai Bộ Tối ưu Hóa (Two-Optimizer)**:
   * Phân tách rạch ròi quá trình tối ưu trong mỗi batch: *Model Step* cập nhật $B_t$ và Head; *Alignment Step* cập nhật duy nhất các ánh xạ $A$ và $D$ trên đặc trưng đã ngắt gradient (`detach()`).
6. **Bộ phân loại Gaussian-Bayes Không Lưu Mẫu**:
   * Mỗi lớp được đại diện bởi kỳ vọng và hiệp phương sai $(\mu_c, \Sigma_c)$.
   * Khi qua tác vụ mới, thống kê lớp cũ được vận chuyển affine qua map $A$: $\mu'_c = \mu_c W_A^\top + b_A$ và $\Sigma'_c = W_A \Sigma_c W_A^\top$. Suy luận phân loại bằng khoảng cách Mahalanobis log-likelihood.

> Chi tiết toán học và báo cáo học thuật đầy đủ được lưu tại:
> * [Báo cáo Nghiên cứu Khoa học (SCIENTIFIC_REPORT.md)](docs/SCIENTIFIC_REPORT.md)
> * [Đặc tả Toán học & Nguyên lý Hoạt động Hướng 1 (DIRECTION1_SPEC.md)](docs/DIRECTION1_SPEC.md)
> * [Sơ đồ Kiến trúc & Luồng Dữ liệu (ARCHITECTURE.md)](docs/ARCHITECTURE.md)

---

## 2. Cấu trúc Dự án

```text
BiCyc_MultiAdapter/
├── configs/                     # Cấu hình thử nghiệm Hydra YAML
│   ├── experiment/              # Presets: keeplora_bicyc, keeplora_bicyc_8gb, keeplora_original...
│   ├── model/                   # Cấu hình ViT-Base & KeepLoRA
│   └── data/                    # Cấu hình 10 tasks CIFAR-100
├── docs/                        # Báo cáo và tài liệu khoa học
│   ├── SCIENTIFIC_REPORT.md     # Báo cáo nghiên cứu học thuật chuẩn bài báo/luận văn
│   ├── DIRECTION1_SPEC.md       # Toàn văn đặc tả toán học chi tiết Hướng 1
│   ├── ARCHITECTURE.md          # Sơ đồ dòng dữ liệu và ranh giới gradient
│   └── RUN_DIRECTION1.md        # Hướng dẫn chi tiết benchmark & ablation
├── scripts/
│   └── smoke_direction1.py      # Smoke test kiểm tra logic toàn diện trong vài giây
├── src/bicyc_multiadapter/      # Mã nguồn chính
│   ├── data/                    # Split CIFAR-100 CIL không lưu mẫu (cil_dataset.py)
│   ├── models/
│   │   ├── backbones/           # ViT đóng băng từ timm (vit_timm.py)
│   │   ├── adapters/            # KeepLoRA SVD (keeplora.py) & PFD Router (routing.py)
│   │   ├── alignment/           # BiCyc, L_iso (bicyc.py) & Channel Gate (distribution.py)
│   │   ├── classifier.py        # GaussianCILClassifier (transport qua A, Bayes scoring)
│   │   └── keeplora_model.py    # Mô hình tích hợp ViT + RoutedKeepLoRA
│   ├── engine/
│   │   ├── keeplora_trainer.py  # Huấn luyện 2-Optimizer tách biệt
│   │   └── task_loop.py         # Pipeline vòng đời CIL, snapshot, resume tự động
│   ├── evaluation/              # Đo lường: accuracy matrix, forgetting, drift
│   └── utils/                   # Checkpoint an toàn, reproducibility, logging
└── tests/unit/                  # Kiểm thử đơn vị cho từng module
```

---

## 3. Cài đặt Môi trường (Windows / Linux / macOS)

### Yêu cầu Hệ thống
* **Python**: **3.11 hoặc 3.12**
* **Phần cứng**: Khuyến nghị GPU NVIDIA $\ge$ 8 GB VRAM (RTX 3060, 4060, 4070... hoặc T4/P100 trên Cloud). Vẫn có thể chạy trên CPU để kiểm thử logic.
* **CUDA**: Driver tương thích CUDA $\ge$ 12.4 (nếu dùng GPU).

### Bước 1: Khởi tạo và kích hoạt môi trường ảo

**Trên Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

**Trên Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### Bước 2: Cài đặt PyTorch & Torchvision

* **Nếu có GPU NVIDIA (CUDA 12.4):**
  ```bash
  pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
  ```
* **Nếu chỉ dùng CPU:**
  ```bash
  pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
  ```

### Bước 3: Cài đặt thư viện phụ thuộc và cài đặt gói dự án
```bash
pip install -r requirements/base.txt
pip install -e .
```
*(Tùy chọn cho nhà phát triển: cài đặt thêm công cụ kiểm thử qua `pip install -r requirements/dev.txt`)*

---

## 4. Hướng dẫn Chạy Thực nghiệm Hướng 1

### 4.1. Kiểm tra nhanh hệ thống (Smoke Test không cần GPU / không cần tải dữ liệu)
Chạy kịch bản kiểm thử toàn bộ vòng đời thuật toán (SVD gradient init, QR projection, 2-optimizer step, channel-wise gate, Gaussian transport) chỉ trong 3–5 giây:
```bash
python scripts/smoke_direction1.py
```
> Kết quả mong đợi: `SMOKE TEST OK`

---

### 4.2. Chạy Thực nghiệm Huấn luyện CIL (10 tasks CIFAR-100)
*(Dữ liệu CIFAR-100 sẽ được tự động tải về thư mục `data/cifar100/` trong lần chạy đầu tiên).*

#### A. Chạy thử 1 epoch/task (Kiểm tra VRAM và luồng dữ liệu trước khi train full):
```powershell
python -m bicyc_multiadapter.train experiment=keeplora_bicyc_8gb experiment.train.epochs_per_task=1 experiment.train.batch_size=16
```

#### B. Huấn luyện Đầy đủ Phương pháp Đề xuất Hướng 1 (Proposed: Multi-Adapter + BiCyc + Channel Gate + $L_{iso}$):
* **Cấu hình tối ưu cho GPU 8 GB (RTX 3060/4060 Desktop/Laptop):**
  ```powershell
  python -m bicyc_multiadapter.train experiment=keeplora_bicyc_8gb
  ```
* **Cấu hình cho GPU lớn ($\ge$ 12 GB VRAM - Batch size 128):**
  ```powershell
  python -m bicyc_multiadapter.train experiment=keeplora_bicyc
  ```

#### C. Huấn luyện Mô hình Đối chứng (Baseline KeepLoRA Nguyên gốc - Gộp adapter, không căn chỉnh):
```powershell
# Cho GPU 8GB:
python -m bicyc_multiadapter.train experiment=keeplora_original_8gb

# Cho GPU >= 12GB:
python -m bicyc_multiadapter.train experiment=keeplora_original
```

#### D. Chạy các Thử nghiệm Triệt tiêu Thành phần (Ablation Studies):
* **Ablation 1: Dùng cổng vô hướng (Scalar Adaptive Gate):**
  ```powershell
  python -m bicyc_multiadapter.train experiment=keeplora_bicyc_8gb experiment.alignment.channelwise_gate=false
  ```
* **Ablation 2: Không sử dụng cổng thích ứng (Cố định $\lambda_t = 1.0$):**
  ```powershell
  python -m bicyc_multiadapter.train experiment=keeplora_bicyc_8gb experiment.alignment.adaptive_gate=false
  ```
* **Ablation 3: Không sử dụng ràng buộc đẳng cự ($L_{iso} = 0$):**
  ```powershell
  python -m bicyc_multiadapter.train experiment=keeplora_bicyc_8gb experiment.alignment.lambda_iso=0.0
  ```

---

### 4.3. Cơ chế Tự động Khôi phục (Auto Resume)
Hệ thống tích hợp sẵn cơ chế lưu checkpoint phân cấp an toàn:
* `checkpoint_boundary.pt`: Tự động lưu trạng thái mô hình ngay khi kết thúc một task.
* `checkpoint_live.pt`: Tự động lưu tiến trình từng epoch và trạng thái optimizer.

Nếu quá trình huấn luyện bị ngắt quãng giữa chừng (mất điện, timeout trên Cloud, nhấn `Ctrl+C`), **chỉ cần chạy lại chính xác câu lệnh huấn luyện trước đó**, hệ thống sẽ tự động phát hiện checkpoint và tiếp tục huấn luyện mà không làm mất dữ liệu đã học.

---

### 4.4. Theo dõi Huấn luyện & Đánh giá Kết quả

* **Xem trực quan hóa biểu đồ qua TensorBoard:**
  ```bash
  tensorboard --logdir outputs
  ```
* **Đánh giá lại mô hình từ checkpoint đã lưu:**
  ```powershell
  python -m bicyc_multiadapter.evaluate experiment=keeplora_bicyc_8gb
  ```
* **Xem số liệu thực nghiệm:**
  Toàn bộ kết quả chi tiết, ma trận độ chính xác $10 \times 10$, mức độ quên (*average forgetting*), độ trôi biểu diễn (*representation drift*) được lưu tại:
  `outputs/<tên_thí_nghiệm>/seed_<seed>/` (gồm các tệp `history.jsonl`, `train_log.csv`, và `run.log`). Bạn có thể sao chép trực tiếp các số liệu này vào biểu mẫu [Báo cáo Nghiên cứu Khoa học (SCIENTIFIC_REPORT.md)](docs/SCIENTIFIC_REPORT.md).

---

### 4.5. Chạy trên Docker hoặc Nền tảng Đám mây (Kaggle / Colab)

* **Chạy bằng Docker:**
  ```bash
  docker compose build
  docker compose run --rm research bash
  
  # Lệnh chạy bên trong container:
  python -m bicyc_multiadapter.train experiment=keeplora_bicyc
  ```
* **Chạy trên Kaggle / Google Colab:**
  Mở notebook [`notebooks/kaggle_train.ipynb`](notebooks/kaggle_train.ipynb), bật chế độ tăng tốc GPU (T4 hoặc P100), chạy tuần tự các cell. Notebook đã tích hợp sẵn AMP `fp16`, TF32 và cell nén kết quả `results.zip` ở cuối phiên.
