# SSM Numerical Stability Guidelines

Selective scan operators in Visual Mamba layers are sensitive to learning rate spikes.
Always enforce:
- Linear warm-up for initial 10 epochs.
- Gradient norm clipping threshold <= 1.0.
