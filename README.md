# BiCyc Multi-Adapter

Scaffold và pipeline nghiên cứu thực nghiệm **Exemplar-Free Class-Incremental Learning (CIL)** cho bài toán học tăng cường không lưu mẫu trên Vision Transformer (ViT):

- **Hướng 1 (`keeplora_bicyc`)**: KeepLoRA nhiều adapter + Adaptive Bidirectional Cycle alignment (BiCyc) + PFD dynamic routing + Adaptive Gaussian-KL Gate (đề xuất mới).
- **Hướng 2 (`rsiat_birae`)**: Shared adapter RSIAT + Bidirectional Residual Autoencoder (Bi-RAE).

---

## Cấu trúc dự án

```text
src/bicyc_multiadapter/
  data/                 # task split, dataset và transform CIL (CIFAR-100)
  models/
    backbones/          # ViT đóng băng (timm)
    adapters/           # KeepLoRA SVD, PFD routing, shared RSIAT
    alignment/          # BiCyc maps, Adaptive Gaussian-KL gate, Bi-RAE
    classifier.py       # Linear Head & Gaussian Bayes Classifier (Transport)
  engine/               # Training loops tách biệt (keeplora_trainer, rsiat_trainer)
  evaluation/           # accuracy, forgetting, drift, metrics
  utils/                # reproducibility, atomic checkpoint, logging
configs/                # Hydra YAML: experiment / model / data
docs/                   # Đặc tả toán học, kiến trúc và tài liệu hướng dẫn
scripts/                # Script smoke test kiểm tra logic nhanh
notebooks/              # Jupyter notebook chạy trên Kaggle / Colab
```

---

## Cài đặt & Kiểm thử trên Local (Windows / Linux / macOS)

Yêu cầu hệ thống:
- **Python**: **3.11 hoặc 3.12**
- **GPU**: Khuyến nghị NVIDIA GPU $\ge$ 8 GB VRAM (RTX 3060/4060/4070... hoặc T4/P100 trên Cloud). Máy không có GPU vẫn có thể chạy chế độ CPU hoặc Smoke test.
- **CUDA**: Driver tương thích CUDA $\ge$ 12.4 (nếu dùng GPU).

### Bước 1: Khởi tạo môi trường ảo

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

**Nếu có NVIDIA GPU (CUDA 12.4):**
```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
```

**Nếu chỉ dùng CPU (hoặc macOS):**
```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
```

### Bước 3: Cài đặt thư viện phụ thuộc và mã nguồn dự án

```bash
pip install -r requirements/base.txt
pip install -e .
```

*(Tùy chọn: Cài thêm công cụ dev/pytest bằng `pip install -r requirements/dev.txt`)*

---

## Hướng dẫn Test & Chạy trên Local

### 1. Kiểm tra nhanh logic (Smoke Test không cần GPU/CIFAR-100)
Chạy script kiểm tra toàn bộ vòng đời (SVD, QR projection, 2 optimizer, Gaussian transport) chỉ trong vài giây:
```bash
python scripts/smoke_direction1.py
# Kết quả mong đợi: "SMOKE TEST OK"
```

### 2. Chạy thử nghiệm Pipeline CIL (10 tasks CIFAR-100)

Dữ liệu CIFAR-100 sẽ được tự động tải về thư mục `data/cifar100/` trong lần chạy đầu tiên.

* **Chạy Pipeline Smoke Test nhanh (1 epoch/task để test VRAM và luồng):**
  ```powershell
  python -m bicyc_multiadapter.train experiment=keeplora_bicyc_8gb experiment.train.epochs_per_task=1 experiment.train.batch_size=16
  ```

* **Chạy huấn luyện đầy đủ Hướng 1 (Proposed: Routed Multi-Adapter + BiCyc + Adaptive Gate):**
  ```powershell
  # Tối ưu cho GPU 8GB (RTX 4060 / 3060 Laptop/Desktop):
  python -m bicyc_multiadapter.train experiment=keeplora_bicyc_8gb

  # Cho GPU lớn (>= 12GB - batch 128):
  python -m bicyc_multiadapter.train experiment=keeplora_bicyc
  ```

* **Chạy Baseline đối chứng KeepLoRA nguyên gốc (Merge adapter, không distillation):**
  ```powershell
  python -m bicyc_multiadapter.train experiment=keeplora_original_8gb
  ```

### 3. Theo dõi & Đánh giá kết quả

* **Xem biểu đồ huấn luyện qua TensorBoard:**
  ```bash
  tensorboard --logdir outputs
  ```
* **Đánh giá lại từ checkpoint:**
  ```powershell
  python -m bicyc_multiadapter.evaluate experiment=keeplora_bicyc_8gb
  ```
* Toàn bộ kết quả, log chi tiết, ma trận độ chính xác và mức độ quên (`forgetting`) được lưu tại: `outputs/<experiment_name>/seed_<seed>/`.

---

## Chạy bằng Docker

```bash
docker compose build
docker compose run --rm research bash

# Bên trong container:
python -m bicyc_multiadapter.train experiment=keeplora_bicyc
```

---

## Chạy trên Cloud (Kaggle / Google Colab)

* **Kaggle**: Mở [`notebooks/kaggle_train.ipynb`](notebooks/kaggle_train.ipynb), bật GPU T4/P100 (đã bật sẵn AMP fp16 + TF32), chạy tuần tự các cell và tải `results.zip` ở cell cuối.
* **Tự động Resume**: Nếu phiên bị ngắt kết nối giữa chừng, chỉ cần chạy lại cùng lệnh với cùng `output_dir`, hệ thống sẽ tự động tiếp tục từ đúng epoch/task gần nhất nhờ `checkpoint_live.pt` và `checkpoint_boundary.pt`.
