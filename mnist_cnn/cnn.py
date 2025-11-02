import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import mnist
from safetensors.torch import save_file

lr=0.001
epochs=10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CNN(nn.Module):
    def __init__(self,num_class,channels,high,weight):
        super().__init__()
        h = high // 4
        w = weight // 4
        linear_1 = int(128 * h * w)
        self.features = nn.Sequential(
            nn.Conv2d(channels,32,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(32,128,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(linear_1,64),
            nn.ReLU(),
            nn.Linear(64,num_class)
        )
    def forward(self,x):
        x = self.features(x)
        x = x.view(x.size(0),-1)
        x = self.classifier(x)
        return x



transform = transforms.Compose([
transforms.ToTensor(),
transforms.Normalize((0.1307,), (0.3081,))  # MNIST 均值与方差
])
print(f"Using device: {device}")

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

model = CNN(num_class=10, channels=1, high=28, weight=28).to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr,weight_decay=1e-4)

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for images,labels in train_loader:
        images,labels = images.to(device),labels.to(device)

        outputs = model(images)
        loss = criterion(outputs,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{epochs}],Loss: {avg_loss:.4f}")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    acc = 100 * correct / total
    print(f"Test Accuracy: {acc:.2f}%")
    save_file(model.state_dict(), f"mnist_cnn-{epoch}.safetensors")
print("finish")

    

model.eval()
test_loss = 0
correct = 0
with torch.no_grad():  # 关闭梯度计算
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss += nn.functional.nll_loss(output, target, reduction='sum').item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
 
test_loss /= len(test_loader.dataset)
acc = 100. * correct / len(test_loader.dataset)
print(f"Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({acc:.2f}%)")
 
 
