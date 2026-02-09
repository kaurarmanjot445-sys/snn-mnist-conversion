
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import pickle
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # as training is slow, use GPU if available,otherwise dafault CPU

# Load MNIST
transform = transforms.Compose([
    transforms.ToTensor(),  
    transforms.Normalize((0.1307,), (0.3081,)) # need to normalize for stable traning
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000) #process in batches 

# Define a flexible MLP that can handle different hidden layer configurations
class FlexibleMLP(nn.Module):
    def __init__(self, hidden_layers):
        #make it flexible by passing list of layer sizes
        super().__init__()
        layers = []
        input_size = 784

     #for each hidden size, add a Linear layer with Xavier init
        for hidden_size in hidden_layers:
            layer = nn.Linear(input_size, hidden_size)
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
            layers.append(layer)
            layers.append(nn.ReLU())
            input_size = hidden_size

        # need raw logits for cross entropy loss
        final_layer = nn.Linear(input_size, 10)
        nn.init.xavier_uniform_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)
        layers.append(final_layer)

        self.network = nn.Sequential(*layers)   #combine all layers together

    
    def forward(self, x):   #how data flows through the network
        x = x.view(x.size(0), -1)
        return self.network(x)

def train_model(model, epochs=30, lr=0.001): # Change epochs from 15 to 30
    model = model.to(device) #move model to GPU if available
    optimizer = optim.Adam(model.parameters(), lr=lr) #optimizer with learning rate
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5) 
    criterion = nn.CrossEntropyLoss() 
    
    for epoch in range(epochs):
        model.train()     # train for up to 30 epochs(set training mode)
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()  #clear old gradients
            output = model(data)
            loss = criterion(output, target)  #calculate loss 
            loss.backward()
            optimizer.step() #update weights 
        
        scheduler.step() 
        test_acc = test_model(model)
        print(f"Epoch {epoch+1} - Test Accuracy: {test_acc:.2f}%") #after each epoch:decay learning rate,test accuracy
        
        if test_acc >= 98.0:
            print(f"Reached 98% accuracy at epoch {epoch+1}") # stop early if 98% reached
            break
    
    return model

#evaluation mode(count correct predictions,no gradients needed)
def test_model(model):
    model.eval()       
    correct = 0
    total = 0
    
    with torch.no_grad():#don't track gradients
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)    
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    accuracy = 100. * correct / total
    return accuracy

# transpose weights to match SNN conversion (need numpy not pytorch)
def extract_weights(model):
    """Extract weights for SNN conversion"""
    weights = []
    biases = []
    
    model = model.cpu()  # move CPU for numpy
    for module in model.network:
        if isinstance(module, nn.Linear):  #check if layer is liner
            W = module.weight.data.numpy().T  
            b = module.bias.data.numpy()
            weights.append(W)
            biases.append(b)
    
    return weights, biases    #need numpy for SNN conversion

# Train all 4 depths
depths_config = {
    '2-layer': [400],
    '4-layer': [400, 400, 400],
    '6-layer': [400, 400, 400, 400, 400],
    '8-layer': [400, 400, 400, 400, 400, 400, 400]
}

results = {}

#train all 4 depths,save weights and biases for SNN conversion.
for name, hidden_layers in depths_config.items():
    print(f"\n{'='*60}")
    print(f"Training {name}: 784 -> {' -> '.join(map(str, hidden_layers))} -> 10")
    print(f"{'='*60}")
    
    model = FlexibleMLP(hidden_layers)
    model = train_model(model, epochs=15, lr=0.001)
    
    # Final test
    final_acc = test_model(model)
    print(f"\nFinal {name} accuracy: {final_acc:.2f}%")
    
    # Save model
    weights, biases = extract_weights(model)
    save_data = {
        'weights': weights,
        'biases': biases,
        'accuracy': final_acc,
        'architecture': [784] + hidden_layers + [10]
    }
    
    filename = f'mnist_{name.replace("-", "_")}.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(save_data, f)
    
    results[name] = final_acc
    print(f"Saved to {filename}")

# Summary
print("\n" + "="*60)
print("TRAINING SUMMARY")
print("="*60)
for name, acc in results.items():
    print(f"{name}: {acc:.2f}%")