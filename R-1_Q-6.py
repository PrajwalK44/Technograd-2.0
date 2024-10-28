!pip install torch torchvision torchsummary

import torch
import torch.nn as nn
import torch.optim as optim
import torch.quantization
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from torchsummary import summary

# Step 1: Load and Preprocess the Data
transform = transforms.Compose([
    transforms.Resize(224),  # ResNet expects 224x224 inputs
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

# Step 2: Load Pretrained ResNet50 Model
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 10)  # CIFAR-10 has 10 classes
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

# Print model summary for reference
print("Original Model Summary:")
summary(model, (3, 224, 224))

# Step 3: Define Training and Evaluation Functions
def train(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

# Step 4: Fine-tune the Model on CIFAR-10 (optional but recommended)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train for a few epochs
epochs = 3
for epoch in range(epochs):
    train_loss = train(model, train_loader, criterion, optimizer, device)
    accuracy = evaluate(model, test_loader, device)
    print(f"Epoch {epoch+1}, Loss: {train_loss:.4f}, Accuracy: {accuracy:.4f}")

# Step 5: Apply Model Compression - Pruning and Quantization
# Pruning the model (layer-by-layer pruning)
from torch.nn.utils import prune

# Apply global pruning
parameters_to_prune = [(module, 'weight') for module in model.modules() if isinstance(module, (nn.Conv2d, nn.Linear))]
prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=0.5)

# Remove the pruning re-parametrization to make pruning permanent
for module, name in parameters_to_prune:
    prune.remove(module, name)

# Quantize the model using dynamic quantization (reduces memory footprint)
model_int8 = torch.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)

# Step 6: Evaluate Compressed Model
compressed_accuracy = evaluate(model_int8, test_loader, device)
print(f"Compressed Model Accuracy: {compressed_accuracy:.4f}")

# Step 7: Save and Check Compressed Model Size
torch.save(model_int8.state_dict(), 'resnet50_compressed.pth')
import os
compressed_size = os.path.getsize('resnet50_compressed.pth') / (1024 * 1024)
print(f"Compressed Model Size: {compressed_size:.2f} MB")

# Final Analysis
print(f"Original Model Accuracy: {accuracy:.4f}")
print(f"Compressed Model Accuracy: {compressed_accuracy:.4f}")
print(f"Memory Reduction Achieved: {compressed_size < 100} (Target: <100 MB)")
