# Checklist hiện thực

## Chung

- [ ] Dataset split CIL có class order cố định và manifest.
- [ ] Backbone ViT/CLIP preload, freeze tuyệt đối và có test `requires_grad=False`.
- [ ] Classifier incremental, checkpoint và resume không lưu raw samples cũ.
- [ ] Seed/determinism/logging metrics theo từng task.

## Hướng 1 — KeepLoRA + Adaptive BiCyc

- [ ] Trích principal subspace bằng SVD từ statistics/gradient được protocol cho phép.
- [ ] Project cập nhật `B` vào residual subspace; `A` LoRA frozen.
- [ ] Old/current forward riêng; old feature detached.
- [ ] Implement MMD hoặc centroid cosine để suy ra `lambda_adaptive` có clamp rõ ràng.
- [ ] Đơn vị test chứng minh alignment optimizer không cập nhật KeepLoRA và ngược lại.

## Hướng 2 — RSIAT + Bi-RAE

- [ ] Shared adapter duy nhất giữ nguyên số parameter sau task 1.
- [ ] RS loss và warm-up `lambda_rs * min(1, epoch / warmup_epochs)`.
- [ ] Bi-RAE forward/backward/cycle loss và orthogonal prototype loss.
- [ ] Eval/inference bypass hoàn toàn Bi-RAE.
