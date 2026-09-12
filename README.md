# BiCyc Multi-Adapter: Exemplar-Free Class-Incremental Learning (EFCIL)

Hệ thống mã nguồn và khung thực nghiệm nghiên cứu bài toán **Học tăng cường theo lớp không lưu mẫu (Exemplar-Free Class-Incremental Learning - EFCIL)** trên mô hình thị giác nền tảng **Vision Transformer (ViT)** đóng băng.

Dự án phát triển kiến trúc lai **`keeplora_bicyc`**: Kết hợp **KeepLoRA** ngân hàng đa adapter, định tuyến phân phối đặc trưng đại diện (**PFD Router**), căn chỉnh phân phối hai chiều (**BiCyc**), cùng hai đóng góp lý thuyết mới: **Cổng phân phối thích ứng theo kênh (Channel-wise KL Gate)** và **Ràng buộc bảo toàn đẳng cự (Isometric Regularizer)**.

---

## 1. Tổng Quan Hướng Nghiên Cứu

### 1.1. Bối Cảnh & Thách Thức
* **Bài toán**: Huấn luyện tuần tự $T$ tác vụ phân loại lớp mới (chuẩn benchmark 10 tasks CIFAR-100, mỗi task 10 lớp phân biệt).
* **Ràng buộc khắt khe**: **Tuyệt đối không lưu trữ bất kỳ ảnh mẫu cũ nào ($0$ exemplars)** nhằm tuân thủ quyền riêng tư (GDPR) và tiết kiệm bộ nhớ phần cứng.
* **Mục tiêu**: Giải quyết triệt để nghịch lý giữa bảo tồn tri thức cũ (*Stability*) và tiếp thu khái niệm mới (*Plasticity*), loại bỏ hiện tượng *quên thảm họa (catastrophic forgetting)*.

### 1.2. Các Đóng Góp Cốt Lõi
1. **KeepLoRA Residual SVD Initialization**: Chiếu gradient vào không gian hạch (Null-space) trực giao với tri thức cũ $(I - Q_{t-1}Q_{t-1}^\top)G_t$, khởi tạo adapter bảo toàn hàm số $\Delta W_t^{(0)} = \mathbf{0}$.
2. **Ngân Hàng Đa Adapter & Định Tuyến PFD (Cosine Router)**: Duy trì các adapter riêng biệt theo từng tác vụ (không gộp đè làm loãng tri thức). Khi suy diễn, tự động kích hoạt adapter tối ưu nhất theo cơ chế Hard Top-1.
3. **Căn Chỉnh Hai Chiều BiCyc & Ràng Buộc Đẳng Cự ($L_{iso}$ - Đề xuất mới)**: Mạng affine đối ngẫu $A$ và $D$ chuyển đổi đặc trưng giữa hai không gian cũ-mới, kết hợp hàm phạt đẳng cự giữ $|\det(W_A)| \approx 1$ chống co rút/bùng nổ elip phân phối.
4. **Cổng Phân Phối Thích Ứng Theo Kênh ($\vec{\lambda}_{t,i}$ - Đề xuất mới)**: Tính toán phân kỳ KL đối xứng trên 768 kênh của ViT. Kênh bảo lưu ngữ nghĩa chung được siết chặt chưng cất ($\lambda \to 1.0$), kênh học mới được nới lỏng ($\lambda \to 0.3$).
5. **Rào Cản Hai Bộ Tối Ưu (Two-Optimizer Barrier)**: Tách biệt hai luồng tối ưu: *Model Step* cập nhật Adapter/Head; *Alignment Step* cập nhật mạng căn chỉnh $A, D$ trên vector đã ngắt gradient (`detach()`).
6. **Vận Chuyển Phân Phối Giải Tích & Phân Loại Bayes**: Lưu trữ thống kê lớp dưới dạng phân phối Gauss $(\mu_c, \Sigma_c)$ dung lượng siêu nhẹ (~6.2KB/lớp). Vận chuyển tức thì qua công thức giải tích khi sang task mới, phân loại bằng khoảng cách Mahalanobis khử thiên vị recency bias.

---

## 2. Minh Họa Các Luồng Hoạt Động Cốt Lõi

Hệ thống vận hành khép kín qua **4 giai đoạn**:

### Giai Đoạn 1: Chiếu Gradient Null-Space & Khởi Tạo KeepLoRA
Gradient tác vụ mới được chiếu qua phần bù trực chuẩn để không làm thay đổi các hướng kích hoạt quan trọng của tác vụ cũ. Khởi tạo $A_t, B_t^{(0)}$ sao cho độ lệch trọng số ban đầu triệt tiêu hoàn toàn.

<p align="center">
  <img src="docs/figures/flow_phase1_init.svg" alt="Giai Đoạn 1: Khởi Tạo KeepLoRA & Chiếu Null-Space" width="90%">
</p>

---

### Giai Đoạn 2: Huấn Luyện 2-Optimizer & Cổng KL 768 Kênh Thích Ứng
Tách biệt hai luồng tối ưu độc lập bằng rào cản `detach()`. Cổng $\vec{\lambda}_{t,i}$ phân hóa linh hoạt 768 kênh ViT dựa trên độ dịch chuyển phân phối Gaussian-KL.

<p align="center">
  <img src="docs/figures/flow_phase2_training.svg" alt="Giai Đoạn 2: Huấn Luyện Two-Optimizer & Cổng Kênh Thích Ứng" width="90%">
</p>

---

### Giai Đoạn 3: Tiêu Hủy Dữ Liệu Thô (GDPR) & Vận Chuyển Gauss Giải Tích
Ngay khi hoàn thành task, toàn bộ ảnh thô bị xóa bỏ. Phân phối Gauss $(\mu_c, \Sigma_c)$ của các lớp cũ được vận chuyển giải tích sang tọa độ mới qua ánh xạ Affine $A$ mà không cần sinh ảnh mẫu giả.

<p align="center">
  <img src="docs/figures/flow_phase3_post_task.svg" alt="Giai Đoạn 3: Tiêu Hủy Ảnh & Vận Chuyển Giải Tích" width="90%">
</p>

---

### Giai Đoạn 4: Định Tuyến Zero-Hint Cosine Router & Phân Loại Mahalanobis
Khi thử nghiệm mẫu ảnh bất kỳ (không cung cấp Task-ID), Cosine Router tìm adapter có độ tương đồng phân phối cao nhất. Vector đặc trưng được chấm điểm log-likelihood trên toàn bộ các lớp qua khoảng cách Mahalanobis elip.

<p align="center">
  <img src="docs/figures/flow_phase4_inference.svg" alt="Giai Đoạn 4: Định Tuyến Suy Diễn & Phân Loại Bayes" width="90%">
</p>

> 🔗 **Tài Liệu Chi Tiết & Demo Tương Tác Trực Tiếp:**
> * [**Báo Cáo Chi Tiết Về Luồng Hoạt Động & Công Thức Toán (WORKFLOW_EXPLAINED.md)**](docs/WORKFLOW_EXPLAINED.md) *(Khuyên đọc)*
> * [**Trang Trình Chiếu Báo Cáo Slide Động (presentation_slides.html)**](docs/presentation_slides.html)
> * [**Phần Mềm Mô Phỏng Trực Quan Pipeline Tương Tác (interactive_pipeline.html)**](docs/interactive_pipeline.html)
> * [Báo Cáo Nghiên Cứu Khoa Học Chuẩn Luận Văn (SCIENTIFIC_REPORT.md)](docs/SCIENTIFIC_REPORT.md)
> * [Đặc Tả Toán Học Chi Tiết Hướng 1 (DIRECTION1_SPEC.md)](docs/DIRECTION1_SPEC.md)

---

## 3. Cài Đặt Môi Trường

### Yêu cầu phần cứng & phần mềm:
* **Python**: 3.11 hoặc 3.12
* **Phần cứng**: GPU NVIDIA $\ge$ 8 GB VRAM (RTX 3060, 4060, 4070... hoặc T4/P100 trên Cloud). Có hỗ trợ chạy CPU để kiểm thử logic.
* **CUDA**: Khuyến nghị CUDA $\ge$ 12.4.

### Các bước cài đặt:

```bash
# 1. Khởi tạo và kích hoạt môi trường ảo
# Trên Windows (PowerShell):
python -m venv .venv
.venv\Scripts\Activate.ps1

# Trên Linux / macOS:
# python3 -m venv .venv && source .venv/bin/activate

# 2. Cài đặt PyTorch & Torchvision (CUDA 12.4)
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
# (Nếu chỉ dùng CPU: pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu)

# 3. Cài đặt các thư viện phụ thuộc và mã nguồn dự án
pip install -r requirements/base.txt
pip install -e .
```

---

## 4. Khởi Tạo & Chạy Thực Nghiệm

### 4.1. Kiểm Tra Nhanh Hệ Thống (Smoke Test - 3 giây)
Kiểm tra toàn bộ luồng thuật toán (SVD gradient, chiếu QR, 2-optimizer, kênh KL gate, vận chuyển Gauss) mà không cần tải dữ liệu và không cần GPU:
```bash
python scripts/smoke_direction1.py
```
> Kết quả chuẩn: `SMOKE TEST OK`

---

### 4.2. Huấn Luyện Toàn Bộ Pipeline 10 Tasks (CIFAR-100)
*(Dữ liệu benchmark CIFAR-100 được tự động tải về thư mục `data/cifar100/` trong lần chạy đầu tiên).*

#### A. Chạy Thử 1 Epoch/Task (Kiểm tra VRAM & pipeline):
```powershell
python -m bicyc_multiadapter.train experiment=keeplora_bicyc_8gb experiment.train.epochs_per_task=1 experiment.train.batch_size=16
```

#### B. Chạy Đầy Đủ Phương Pháp Đề Xuất (Proposed: Multi-Adapter + BiCyc + Channel Gate + $L_{iso}$):
```powershell
# Dành cho GPU 8 GB VRAM (RTX 3060/4060/Laptop):
python -m bicyc_multiadapter.train experiment=keeplora_bicyc_8gb

# Dành cho GPU lớn (>= 12 GB VRAM - Batch size 128):
python -m bicyc_multiadapter.train experiment=keeplora_bicyc
```

#### C. Chạy Mô Hình Đối Chứng (Baseline KeepLoRA Gốc - Gộp adapter, không căn chỉnh BiCyc):
```powershell
# Dành cho GPU 8 GB:
python -m bicyc_multiadapter.train experiment=keeplora_original_8gb

# Dành cho GPU >= 12 GB:
python -m bicyc_multiadapter.train experiment=keeplora_original
```

#### D. Chạy Các Thử Nghiệm Triệt Tiêu (Ablation Studies):
```powershell
# 1. Dùng cổng vô hướng (Scalar Adaptive Gate):
python -m bicyc_multiadapter.train experiment=keeplora_bicyc_8gb experiment.alignment.channelwise_gate=false

# 2. Không dùng cổng thích ứng (Cố định lambda = 1.0):
python -m bicyc_multiadapter.train experiment=keeplora_bicyc_8gb experiment.alignment.adaptive_gate=false

# 3. Không dùng ràng buộc đẳng cự (L_iso = 0):
python -m bicyc_multiadapter.train experiment=keeplora_bicyc_8gb experiment.alignment.lambda_iso=0.0
```

---

### 4.3. Cơ Chế Tự Động Khôi Phục (Auto Resume)
Hệ thống tự động ghi nhớ trạng thái huấn luyện:
* `checkpoint_boundary.pt`: Tự động lưu khi vừa kết thúc một task.
* `checkpoint_live.pt`: Tự động lưu từng epoch và trạng thái optimizer.

Nếu quá trình huấn luyện bị gián đoạn (mất điện, ngắt kết nối Cloud, `Ctrl+C`), **chỉ cần chạy lại chính xác câu lệnh trước đó**, mã nguồn sẽ tự động phát hiện và huấn luyện tiếp từ điểm dừng.

---

### 4.4. Theo Dõi & Đánh Giá Kết Quả

* **Mở bảng điều khiển trực quan hóa TensorBoard:**
  ```bash
  tensorboard --logdir outputs
  ```
* **Đánh giá lại mô hình từ checkpoint đã lưu:**
  ```powershell
  python -m bicyc_multiadapter.evaluate experiment=keeplora_bicyc_8gb
  ```
* **Vị trí lưu trữ dữ liệu thực nghiệm:**
  Ma trận độ chính xác $10 \times 10$, mức độ quên trung bình (*average forgetting*), độ trôi biểu diễn (*representation drift*) được ghi tự động tại:
  `outputs/<tên_thí_nghiệm>/seed_<seed>/` (gồm `history.jsonl`, `train_log.csv`, và `run.log`).

---

### 4.5. Triển Khai Docker & Cloud (Kaggle / Colab)
* **Docker:**
  ```bash
  docker compose build
  docker compose run --rm research bash
  python -m bicyc_multiadapter.train experiment=keeplora_bicyc
  ```
* **Kaggle / Colab:** Sử dụng notebook [`notebooks/kaggle_train.ipynb`](notebooks/kaggle_train.ipynb), bật chế độ GPU (T4/P100). Notebook đã cấu hình sẵn AMP `fp16`, TF32 và tự động nén `results.zip` sau khi hoàn thành.
