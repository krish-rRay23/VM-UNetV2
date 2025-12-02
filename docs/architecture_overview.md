# VM-UNetV2 Architecture Overview

## Visual State Space Models (VSSM)
VM-UNetV2 integrates 2D-Selective-Scan (SS2D) Visual State Space Blocks into a UNet-like encoder-decoder topology.
Unlike self-attention mechanisms with quadratic computational complexity $O(N^2)$, the SSM formulation achieves global receptive field with linear complexity $O(N)$.

## Core Components
1. **Patch Partition & Linear Embedding**: Maps raw medical images into token embeddings.
2. **VSS Blocks**: Cross-scan bidirectional SSM layers capturing spatial correlations.
3. **Skip Connections**: Concatenates multi-scale shallow features to retain high-frequency boundary details.
4. **Decoder**: Progressive bilinear upsampling with residual refinement.
