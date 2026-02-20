# ReLU to SNN Conversion on MNIST
we are converting trained ReLU networks into Spiking Neural Networks — no retraining needed, just math. Based on Stanojevic et al., *Nature Communications* 2024, with guidance from Dr. Guillaume Bellec (TU Graz, Austria).

---
## What I did

- Trained ReLU networks (2, 4, 6, 8 hidden layers) on MNIST
- Converted each to a TTFS SNN using the B1 identity mapping (Eq. 9)
- Verified numerically that SNN == ReLU at every layer

---
## Results

| Hidden Layers | ReLU Acc | SNN Acc | Sparsity |
|--------------:|:--------:|:-------:|:--------:|
| 2 | 98.20% | 98.20% | 0.44 |
| 4 | 98.32% | 98.32% | 0.26 |
| 6 | 98.23% | 98.23% | 0.25 |
| 8 | 98.26% | 98.26% | 0.20 |

0% accuracy drop —correct,not a bug.The conversion is mathematically exact.

---
## Figures

### Accuracy and Sparsity vs Depth
![this is a image] ([https://github.com/kaurarmanjot445-sys/snn-mnist-conversion/blob/main/depth_comparison.png?raw=true)](https://github.com/kaurarmanjot445-sys/snn-mnist-conversion/blob/af78c53f95a53014bd1ec49abe9bac180545c1a3/depth_comparison.png))

### Numerical Stability: Correct vs Wrong t_max
![Stability Analysis]

---

## How it works
Each neuron fires one spike at time `t`. Activation is recovered as `x = t_max - t`.
B1 mapping sets:
- `W_snn = W_relu` (weights unchanged)
- `V = (t_max - t_min) - b` (threshold from bias)
Timing chains across layers: `t_min` of layer n = `t_max` of layer n-1.

## Files

| File | What it does |
|------|-------------|
| `single_layer_demo.py` | Verify math on toy networks first |
| `train_mnist_pytorch.py` | Train and save ReLU models |
| `convert_all_models.py` | Convert to SNN, check accuracy |
| `create_plot.py` | Generate figures |
| `stability_analysis.py` | Show why correct t_max matters |

## Run

```python single_layer_demo.py
python train_mnist_pytorch.py
python convert_all_models.py
python create_plot.py
```
## Acknowledgements

- Dr. Guillaume Bellec guidance

