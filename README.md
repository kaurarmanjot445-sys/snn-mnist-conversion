ReLU-to-SNN Conversion: Depth Analysis
Systematic study of how network depth affects time-to-first-spike conversion quality on MNIST.

Results
Depth Comparison
![Depth Comparison] ([depth_comparison.png](https://1drv.ms/i/c/5ad32c5f1e1dc822/IQD7CMmkCI9aTLsh3UTJ16sDAZ11Dk1pKGtSzOB1NA2dln8?e=CfQy4P))

## Depth Comparison

| Layers | ReLU Acc | SNN Acc | Drop |
|--------|----------|---------|------|
| 2      | 96.70%   | 96.50%  | 0.20% |
| 4      | 96.30%   | 95.60%  | 0.70% |
| 6      | 96.50%   | 93.10%  | 3.40% |
| 8      | 96.40%   | 91.20%  | 5.20% |

Conversion maintains accuracy for shallow networks but degrades with depth due to spike-time clustering near temporal boundaries.

Numerical Verification
2-layer precision: 2.47 × 10⁻⁹ error (machine precision confirmed)

Method
Architecture: 784 → [400 × N] → 10 (N = 1, 3, 5, 7 hidden layers)

Training: Adam (lr=0.001), batch size 64, Xavier init, 30 epochs max

Conversion:
t_max = 15.0
V_threshold = t_max - bias - Σ(weights)
spike_time = V_threshold / input_current
output = t_max - spike_time


## Files

-train_mnist_pytorch.py - Train 2, 4, 6, 8-layer ReLU networks on MNIST
-convert_all_models.py - Convert trained models to SNNs and evaluate
-create_plot.py - Generate depth vs accuracy comparison plot
-numerical_exactness_test.py - Verify conversion precision on 2-layer network

**Figures:**
- depth_comparison.png - Main result showing accuracy degradation with depth
- spike_time_distributions.png - Histogram showing spike clustering in deep networks
- time_axis_schematic.png - Temporal propagation diagram across layers 

Requirements  
pip install numpy torch torchvision matplotlib  

Key Insight  
Exact gradient equivalence does not guarantee numerical stability. Precision limits emerge at depth when spike times cluster.  

Acknowledgments  
Developed under the guidance of Prof. Guillaume Bellec.




