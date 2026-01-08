# Visual State Space (VSS) Block Specification

## 2D Selective Scan (SS2D)
Because visual data is non-sequential and 2-dimensional, standard 1D selective scan lacks bidirectional spatial context.
SS2D scans visual features in 4 distinct directions:
1. Top-left to bottom-right (Row-wise forward)
2. Bottom-right to top-left (Row-wise reverse)
3. Top-left to bottom-right (Column-wise forward)
4. Bottom-right to top-left (Column-wise reverse)

All 4 sequences pass through independent S6 SSM layers, and are subsequently merged and projected back to the spatial token plane.
