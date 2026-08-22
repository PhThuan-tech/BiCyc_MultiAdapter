# Quy ước tái lập thí nghiệm

## Ma trận môi trường cần ghi nhận

Mỗi run phải ghi vào `outputs/<run-id>/environment.json`: Git commit, image digest, Python/PyTorch/CUDA, GPU, config đã resolve và seed. Khi viết runner, không dùng bản config mặc định ngầm định.

| Thành phần | Chuẩn ban đầu |
| --- | --- |
| Python | 3.11.x |
| PyTorch / torchvision | 2.5.1 / 0.20.1 |
| CUDA runtime trong image | 12.4 + cuDNN 9 |
| Điều phối config | Hydra 1.3.2 |
| Seeds tối thiểu | 3: 2024, 2025, 2026 |

## Quy tắc dữ liệu và CIL

- Không lưu raw exemplar của task cũ. Chỉ checkpoint, prototype/statistics mà protocol cho phép.
- Lưu file task split, thứ tự class và transform trong thư mục run; không tái sinh chúng ở lần đánh giá.
- Tất cả ablation dùng cùng split, backbone, budget epoch, optimizer và seed.

## Quy tắc gradient

- `old_model` là snapshot `eval()` và toàn bộ tensor feature cũ phải `detach()` / `torch.no_grad()`.
- Với KeepLoRA + BiCyc, tách bước tối ưu alignment `(A, D)` khỏi loss cập nhật LoRA; xác nhận bằng test rằng gradient không về old model và các ma trận bị freeze.
- Bi-RAE chỉ tồn tại trong training; evaluator/inference không được khởi tạo hoặc gọi nó.

## Docker ổn định

Giữ tag base image cùng với digest khi bắt đầu benchmark chính thức (`docker image inspect ...`). Khi đổi PyTorch/CUDA, tạo một series benchmark mới thay vì trộn kết quả. Mount dữ liệu/checkpoint qua volume, không bake vào layer Docker. Dùng `docker compose run --rm`, không chạy container có state lâu dài.
