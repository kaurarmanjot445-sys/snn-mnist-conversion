ReLU-to-SNN Conversion: Depth Analysis

This repository studies how converting deep ReLU networks to spiking neural networks affects accuracy and temporal spike dynamics on MNIST

**Results**

Depth Comparison
https://github.com/kaurarmanjot445-sys/snn-mnist-conversion/blob/main/depth_comparison.png?raw=true



| Layers | ReLU Acc | SNN Acc | Drop 
|--------|----------|---------|------|
| 2      | 96.10%   | 96.40%  | -0.30% |
| 4      | 94.20%   | 95.60%  | 1.20% |
| 6      | 96.10%   | 93.10%  | 1.60% |
| 8      | 96.80%   | 91.20%  | 3.30% |

Conversion maintains accuracy for shallow networks but degrades with depth due to spike-time clustering near temporal boundaries.

## Numerical Verification

Tested conversion precision across network depths (2-100 layers):
- **Correct parameters:** Error < 10⁻¹⁴ (machine precision) at all depths
- **5% threshold error:** Breaks immediately (error ~1.0)

Demonstrates formula correctness and sensitivity to parameter precision.

[(Stability_Analysis.png)](https://github.com/kaurarmanjot445-sys/snn-mnist-conversion/blob/main/stability_analysis.png?raw=true)

 **Method:**
 
**Architecture:** 784 → [400 × N] → 10 (N = 1, 3, 5, 7 hidden layers)

**Training:** Adam (lr=0.001), batch size 64, Xavier init, 30 epochs max

Conversion:

```
t_max = 15.0
V_threshold = t_max - bias - Σ(weights)
spike_time = V_threshold / input_current
output = t_max - spike_time
```

## Files Usage

1. **Train networks**  
   `python train_mnist_pytorch.py`  
   Train 2, 4, 6, 8-layer ReLU networks on MNIST.
2. **Convert trained models to SNN**  
   `python convert_all_models.py`  
   Convert trained models to SNNs and evaluate.
3. **Generate plots**  
   `python create_plot.py`  
   Generate depth vs accuracy comparison plot.
4. **Verify numerical exactness**  
   `python stability_analysis.py`  
   Verify conversion precision on 2-layer network.


**Figures:**
1. depth_comparison.png - Main result showing accuracy degradation with depth
2. stability_analysis.png 
   

**Requirements:**
pip install numpy torch torchvision matplotlib  

**Key Insight**  
Exact gradient equivalence does not guarantee numerical stability. Precision limits emerge at depth when spike times cluster.  

**Acknowledgments:** 
Developed under the guidance of Prof. Guillaume Bellec.




