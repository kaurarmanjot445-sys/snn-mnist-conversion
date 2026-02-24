# ReLU to SNN Conversion on MNIST
This project converts trained ReLU networks into Time-to-First-Spike (TTFS) Spiking Neural Networks using the B1 identity mapping from Stanojevic et al., Nature Communications 2024.

---
## What I did

- Trained ReLU networks (2, 4, 6, 8 hidden layers) on MNIST
- Converted each to a TTFS SNN using the B1 identity mapping (Eq. 9)
- Verified numerically that SNN == ReLU at every layer

---
## Results

|Hidden Layers| ReLU Acc | SNN Acc | Sparsity |
|------------:|:--------:|:-------:|:--------:|
|2 | 98.26% | 98.26% | 0.54 |
|4 | 98.24% | 98.24% | 0.30 |
|6 | 98.23% | 98.23% | 0.27 |
|8 | 98.25% | 98.25% | 0.21 |

The conversion preserves accuracy, consistent with the exact theoretical mapping described in Eq. 9 of the paper.

---
## Figures

### Accuracy and Sparsity vs Depth
![Accuracy and sparsity vs depth](https://github.com/kaurarmanjot445-sys/snn-mnist-conversion/blob/main/depth_comparison.png?raw=true)

### Numerical Stability: Correct vs Wrong t_max
![Numerical error: correct vs wrong t_max](https://github.com/kaurarmanjot445-sys/snn-mnist-conversion/blob/fcaf742c87fa216e0bd425dd7d5064f837f83b31/stability_analysis.png)

---

## How it works
The conversion was verified by comparing layer-wise activations and final classification accuracy between the original ReLU network and the converted SNN. Each neuron fires one spike at time 't'. Activation is recovered as: 'x = (t_max - t) / τ_c (τ_c = 1 here, so x = t_max - t in code)'
B1 mapping sets(Eq. 9):
- `W_snn = W_relu` (weights unchanged)
- `V = (t_max - t_min)/τ_c - b` (τ_c = 1 throughout, so V = t_max - t_min - b in code`
Timing chains across layers: `t_min` of layer n = `t_max` of layer n-1.
**Why B1 matters:** Guarantees gradient descent follows identical training trajectories to the ReLU network. Other models (like α1) still diverge during training even with smart initialization — B1 fixes that.

**Why t_max matters:** Must fit actual activation range at each layer.Too small to clipping to errors 1.0(approx). Correct derivation to machine precision 1e-15(approx) (see stability plot).

## Files

| File | What it does |
|------|-------------|
| `single_layer.py` | Verify math on toy networks first |
| `train_mnist_pytorch.py` | Train and save ReLU models |
| `convert_all_models.py` | Convert to SNN, check accuracy |
| `create_plot.py` | Generate figures |
| `stability_analysis.py` | Show why correct t_max matters |

## Run
```
python single_layer.py
python train_mnist_pytorch.py
python convert_all_models.py
python create_plot.py
python stability_analysis.py
```
## Requirements
```torch>=1.12
torchvision>=0.13
numpy>=1.21
matplotlib>=3.5
```
## Acknowledgements
- Guidance provided by Dr.Guillaume Bellec

