
import numpy as np
import pickle

# load MNIST data
def load_mnist():
    """Load MNIST using keras (lighter than torchvision,torchvision was not working so... )"""
    from tensorflow import keras
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Normalize to [0, 1]
    x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 784).astype('float32') / 255.0
    
    return x_train, y_train, x_test, y_test  #as neural networks work better with normalized inputs

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

class SimpleMLP:
    def __init__(self):
        # Initialize weights (Xavier initialization)
        self.W1 = np.random.randn(784, 400) * np.sqrt(2.0 / 784)
        self.b1 = np.zeros(400)
        self.W2 = np.random.randn(400, 400) * np.sqrt(2.0 / 400)
        self.b2 = np.zeros(400)
        self.W3 = np.random.randn(400, 10) * np.sqrt(2.0 / 400)  #xavier /he imtialization
        self.b3 = np.zeros(10)  #training will adjust all to minimize mistake
    
    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1  # combine all with weights
        self.h1 = relu(self.z1) # keep positives zeros out neg
        self.z2 = self.h1 @ self.W2 + self.b2 
        self.h2 = relu(self.z2)  #layer2 combines them into complex patters
        self.z3 = self.h2 @ self.W3 + self.b3  #output layer
        output = softmax(self.z3)  #covert to probabilities
        return output  #(we are passing through 3 layers each doing linear transform-relu)
    
    def backward(self, X, y, output, lr=0.001):
        batch_size = X.shape[0]   
        
        # Output layer gradient
        dz3 = output.copy()
        dz3[range(batch_size), y] -= 1
        dz3 /= batch_size    # calculating how wrong we are ,then we will trace it back to layers,adjust weights
        
        dW3 = self.h2.T @ dz3
        db3 = np.sum(dz3, axis=0)
        
        # Hidden layer 2 gradient
        dh2 = dz3 @ self.W3.T
        dz2 = dh2 * (self.z2 > 0)
        dW2 = self.h1.T @ dz2
        db2 = np.sum(dz2, axis=0)
        
        # Hidden layer 1 gradient
        dh1 = dz2 @ self.W2.T
        dz1 = dh1 * (self.z1 > 0)
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0)
        
        # Update weights
        self.W3 -= lr * dW3
        self.b3 -= lr * db3
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1  #w3is now slightly diff,will predict better next time
    
    def train(self, X_train, y_train, epochs=3, batch_size=64, lr=0.001):
        n_samples = X_train.shape[0]
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # Forward and backward
                output = self.forward(X_batch)
                self.backward(X_batch, y_batch, output, lr)
                
                if i % 6400 == 0:
                    loss = -np.mean(np.log(output[range(len(y_batch)), y_batch] + 1e-8))
                    print(f'Epoch {epoch+1}/{epochs}, Batch {i//batch_size}, Loss: {loss:.4f}')
    
    def predict(self, X):
        output = self.forward(X)
        return np.argmax(output, axis=1)
    
    def accuracy(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y) * 100

# Load data
print("Loading MNIST...")
X_train, y_train, X_test, y_test = load_mnist()
print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")

# Train model
print("\nTraining model...")
model = SimpleMLP()
model.train(X_train, y_train, epochs=3, batch_size=64, lr=0.001)

# Test accuracy
train_acc = model.accuracy(X_train[:1000], y_train[:1000])
test_acc = model.accuracy(X_test, y_test)
print(f"\nTrain Accuracy (1000 samples): {train_acc:.2f}%")
print(f"Test Accuracy: {test_acc:.2f}%")

# Save weights
weights = {
    'W1': model.W1, 'b1': model.b1,
    'W2': model.W2, 'b2': model.b2,
    'W3': model.W3, 'b3': model.b3
}
with open('mnist_weights.pkl', 'wb') as f:
    pickle.dump(weights, f)
print("\nWeights saved to 'mnist_weights.pkl'")