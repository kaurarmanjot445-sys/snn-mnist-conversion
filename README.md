
*SNN-MNIST Conversion*
Time-to-first-spike spiking neural network conversion for MNIST digit classification, based on High-performance deep spiking neural networks with 0.3 spikes per neuron, published on August 9, 2024,in nature Communications, by Ana Stanojevic. 

# Overview
This project demonstrates conversion of trained ReLU networks to spike-time coded SNNs without retraining. We investigate how network depth affects conversion quality and identify the numerical limits of time-based encoding.

# Key Results
Training Performance
All networks achieved approximately 98% test accuracy:
Architecture Test Accuracy

2-layer (784 to 400  to 10) 98.04%
4-layer (784 to 400^3 to 10) 98.37%
6-layer (784 to 400^5 to 10) 98.06%
8-layer (784 to 400^7 to 10) 98.01%


# Conversion Results
Layers ReLU Acc SNN Acc Drop

2 96.80% 96.50% 0.30%
4 96.80% 96.30% 0.50%
6 96.80% 94.00% 2.80%
8 96.80% 92.00% 4.80%

# Key Finding: Conversion maintains high fidelity for shallow networks (2-4 layers) but shows progressive degradation with depth, exposing numerical precision challenges in multi-layer spike-time propagation.

**Numerical Exactness Verification*
Single-layer test demonstrates machine-precision equivalence:
1.Maximum output difference: 0.0000000247
2.Relative error: 2.47 × 10⁻⁹
# This confirms the mathematical correctness of the conversion formula for shallow architectures.

# Spike-Time Analysis
Spike-time distributions reveal the core challenge:

1. 2-layer: Well-distributed spike times across the temporal window
2. 4-layer: Beginning to cluster
3. 6-layer: Moderate clustering near temporal boundaries
4. 8-layer: Severe clustering near t_max


**When spike times cluster tightly, small numerical errors cause large changes in output activities, leading to misclassification.*

# Method
Conversion Framework
The conversion uses exact gradient equivalence to map ReLU parameters to SNN parameters:
**Time encoding:*
1.High activity to Early spike time

2.Low activity to Late spike time


# Parameter mapping:
V_threshold = t_max - bias - Σ(weights)
α = 1 - Σ(weights)
spike_time = (V_threshold - V_rest) / input_current
Critical detail: The bias term must be included in the threshold calculation. Omitting it breaks the gradient equivalence.

**Time Window Configuration*
Current approach: Fixed t_max = 15.0
Activities encoded in window [0, t_max]

Simpler implementation for validation
Standard in time-to-first-spike literature


# Alternative approach (future work):
1. Fix t_min instead of t_max
2. Derive t_max per layer: t_max(L) = t_min(L+1)
3. Ensures temporal continuity across layers
4.May improve stability for deep networks




# Training Details
1.Optimizer: Adam (learning rate: 0.001)
2. Learning rate schedule: StepLR (decay every 10 epochs, gamma=0.5)
3. Epochs: 30 maximum (early stopping at 98%)
4. Batch size: 64
5. Initialization: Xavier uniform for weights, zeros for biases
6.Framework: PyTorch


# Analysis
*What Works*

# Shallow networks (2-4 layers):
1.Conversion maintains near-perfect accuracy (<1% drop)
2.Numerical precision remains stable
3.Theoretical equivalence holds in practice


# Mathematical framework:
Exact gradient equivalence verified numerically
Single-layer conversion achieves machine precision


# Known Limitations
*Deep networks (6-8 layers):*
1.Progressive accuracy degradation with depth
2.Root cause:Spike-time clustering near temporal boundaries
3.Small floating-point errors to large output changes


# The fundamental challenge: While the conversion is mathematically exact, numerical precision limits emerge when spike times propagate through many layers.Activities near zero map to times near t_max,where the temporal resolution becomes insufficient to distinguish between different activation levels.

# Research Context
This depth-dependent degradation is a known challenge in rate-to-spike conversion methods and represents an active area of research in neuromorphic computing. Potential solutions include:
1.Layer-wise normalization techniques
2.Adaptive temporal windows per layer
3.Hybrid approaches combining conversion with fine-tuning


Repository Structure

1. train_mnist_pytorch.py # Train variable-depth ReLU MLPs
2.  convert_all_models.py # Batch SNN conversion & testing
3. create_plot.py # Generate depth comparison plot
4. numerical_exactness_test.py # Verify conversion precision
5. spike_distribution_plot.py # Visualize spike-time clustering
6. time_axis_schematic.py # Create temporal propagation diagram
7. depth_comparison.png # Main results visualization
8. spike_time_distributions.png # Clustering analysis
9. time_axis_schematic.png # Layer timing diagram
10.  README.md

# Usage
1. Train Models
python train_mnist_pytorch.py  
Trains 2, 4, 6, and 8-layer networks. Saves trained weights to mnist_*_layer.pkl.  
2. Convert and Test  
python convert_all_models.py  
Converts all trained models to SNNs and tests on 1000 MNIST samples.  
3. Generate Visualizations  
python create_plot.py # Depth comparison  
python spike_distribution_plot.py # Spike clustering  
python time_axis_schematic.py # Temporal diagram  
4. Verify Numerical Exactness  
python numerical_exactness_test.py  
Tests conversion precision on 2-layer network.

# Requirements  
pip install numpy torch torchvision matplotlib  
Tested with:  
Python 3.8+  
PyTorch 1.9+  
NumPy 1.21+  
Matplotlib 3.4+  

# Key Insight  
**This work demonstrates that exact gradient equivalence does not guarantee numerical stability. While the conversion is mathematically exact, floating-point precision limitations emerge at depth, exposing a fundamental challenge in time-to-first-spike conversion methods.*  

# Future Directions  
1.Implement layer normalization for improved deep network conversion  
2.Test adaptive t_max strategies that scale with network depth  
3.Explore alternative spike encoding schemes (rate coding, temporal patterns)  
4.Benchmark against direct SNN training methods  
5.Investigate hybrid conversion + fine-tuning approaches  

# References  
Implementation based on:  
High-performance deep spiking neural networks with 0.3 spikes per neuron, published on August 9, 2024,by Ana Stanojevic.   
Exact gradient equivalence framework for rate-to-spike conversion  

# Acknowledgments  
This work was developed under the guidance of Prof. Guillaume Bellec. Special thanks for the insights on numerical stability and time-based encoding challenges.  

