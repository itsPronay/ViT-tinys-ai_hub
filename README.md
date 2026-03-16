# ViT Tiny Models Benchmark (QAI Hub)

This project benchmarks selected `timm` vision models by:
- creating a PyTorch model,
- tracing it,
- submitting a compile job to QAI Hub,
- profiling it on a target device,
- optionally logging results to Weights & Biases.

## Models Supported
- `vit_tiny_patch16_224`
- `mobilevitv2_100`
- `mobilevitv2_125`
- `tiny_vit_5m_224`

## 1) QAI Hub Authentication

Before running benchmarks, make sure your QAI Hub credentials are configured for this machine/account.

If needed, follow the official QAI Hub auth/setup steps for your account.

## 2) Run All Models

Use the provided script [runner.sh](runner.sh):

```bash
bash runner.sh
```

This runs all supported models with:
- image sizes `224` and `448`
- device `Samsung Galaxy S25 (Family)`
- W&B mode `online`


