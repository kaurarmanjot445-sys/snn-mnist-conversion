

import numpy as np
import pickle
from tensorflow import keras

def relu(x):
    return np.maximum(0, x)  #if neg make0, if positive keep same
 
 #simulate when neurons spike
def spiking_simulator(t0, J, alpha, V):
    t1 = (V + t0 @ J) / (alpha + J.sum(0))
    return t1

def simple_mapping(X, W, b, t_max=15.0):
    X_min, X_max = np.min(X), np.max(X)
    if X_max - X_min > 1e-8:  #check of range is meaningful(avoid divide by zero)
        X_norm = (X - X_min) / (X_max - X_min)
        scale = X_max - X_min
    else:
        X_norm = X
        scale = 1.0
    
    # convert activities to spike times
    t_in = 1 - X_norm   #high activity =early time
    J = W
    V = t_max - b - J.sum(0)  #threshold voltage
    alpha = 1.0 - J.sum(0)  # leak term for spike timing
    
    return alpha, J, V, t_max, t_in, scale

#process layer by layer convert activities to spike times to back to activities.
def snn_forward_multi_layer(X, weights, biases):
    """Forward pass through arbitrary depth SNN"""
    h = X  #input activities
    
    for W, b in zip(weights, biases):
        alpha, J, V, tmax, t_in, scale = simple_mapping(h, W, b)
        t_out = spiking_simulator(t_in, J, alpha, V)
        t_out = np.minimum(t_out, tmax)  #clip spikes to valid time range
        h = (tmax - t_out) * scale  #convert spike time back to activities
    
    return h

# standard neural network:;linear transform,then relu,skip relu on last layer
def relu_forward_multi_layer(X, weights, biases):
    """Forward pass through arbitrary depth ReLU network"""
    h = X
    
    for i, (W, b) in enumerate(zip(weights, biases)):
        h = h @ W + b
        if i < len(weights) - 1: # Don't apply ReLU on last layer
            h = relu(h)  #apply activation(only for hidden layers)
    
    return h

# Load MNIST test data
(_, _), (x_test, y_test) = keras.datasets.mnist.load_data()
x_test = x_test.reshape(-1, 784).astype('float32') / 255.0

# Test all models
model_files = [      # trained models i saved earlier
    'mnist_2_layer.pkl',
    'mnist_4_layer.pkl',
    'mnist_6_layer.pkl',
    'mnist_8_layer.pkl'
]

all_results = {}

for model_file in model_files:
    print(f"\n{'='*60}")
    print(f"Testing: {model_file}")
    print(f"{'='*60}")
    
    # Load model, all the parameters from trained model
    with open(model_file, 'rb') as f:
        model_data = pickle.load(f)
    
    weights = model_data['weights']
    biases = model_data['biases']
    n_layers = len(model_data['weights']) # Count weight matrices
    trained_acc = model_data.get('accuracy', 0) #get training accuracy,deafault 0 if missing
    
    #show what i am testing
    print(f"Trained accuracy: {trained_acc:.2f}%")
    print(f"Number of layers: {n_layers}")
    print(f"Testing on 10,000 samples...")
    
    # Test on subset first (faster)
    n_test = 1000
    correct_relu = 0  #counter for relu correct predictions
    correct_snn = 0
    
    for i in range(n_test):
        X = x_test[i:i+1]  #process one image at a time
        y_true = y_test[i]

      # run both relu and snn forward passes on same image,pick class with highest output score   
        # ReLU prediction
        out_relu = relu_forward_multi_layer(X, weights, biases)
        pred_relu = np.argmax(out_relu)   #position of highest score
        
        # SNN prediction
        out_snn = snn_forward_multi_layer(X, weights, biases)
        pred_snn = np.argmax(out_snn)   
        
   # show progrees and count correct predictions for both models
        if pred_relu == y_true:  
            correct_relu += 1
        if pred_snn == y_true:
            correct_snn += 1
        
        if (i+1) % 200 == 0:
            print(f" Tested {i+1}/{n_test}")  
    
    # calculate accuracies and drop in performance
    acc_relu = 100.0 * correct_relu / n_test
    acc_snn = 100.0 * correct_snn / n_test
    
    print(f"\nResults (1000 samples):")
    print(f" ReLU: {acc_relu:.2f}%")
    print(f" SNN: {acc_snn:.2f}%")
    print(f" Drop: {acc_relu - acc_snn:.2f}%")
    
    all_results[n_layers] = {
        'relu_acc': acc_relu,
        'snn_acc': acc_snn,
        'drop': acc_relu - acc_snn
    }

# Save results
with open('conversion_results.pkl', 'wb') as f:
    pickle.dump(all_results, f)

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"{'Layers':<10} {'ReLU Acc':<12} {'SNN Acc':<12} {'Drop':<10}")
print("-" * 50)
for n_layers in sorted(all_results.keys()):
    res = all_results[n_layers]
    print(f"{n_layers:<10} {res['relu_acc']:<12.2f} {res['snn_acc']:<12.2f} {res['drop']:<10.2f}")

print("\nResults saved to 'conversion_results.pkl'")