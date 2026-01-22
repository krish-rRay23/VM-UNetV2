# GPU Memory Optimization Guide

## VRAM Requirements by Input Resolution
| Image Size | Batch Size | Mixed Precision (FP16) | Peak VRAM |
|---|---|---|---|
| 256x256 | 32 | Yes | 7.8 GB (RTX 3080 / 4070) |
| 256x256 | 64 | Yes | 14.2 GB (RTX 4090 / A5000) |
| 512x512 | 16 | Yes | 18.5 GB (A100 / RTX 6000) |

## Activation Checkpointing
Enable gradient checkpointing in `models/vmunet/vmunet_v2.py` to reduce peak activation memory by ~42% during backward pass.
