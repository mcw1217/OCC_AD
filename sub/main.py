import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from copy import deepcopy

def load_data(data_dir, batch_size, img_size):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    train_dataset = datasets.ImageFolder(root=f"{data_dir}/train", transform=transform)
    test_dataset = datasets.ImageFolder(root=f"{data_dir}/test", transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


class DinoBinaryClassifier(nn.Module):
    def __init__(self, pretrained_model):
        super(DinoBinaryClassifier, self).__init__()
        self.dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.classifier = nn.Sequential(nn.Linear(384,1))
        
    def forward(self,x):
        x = self.dino(x)
        x = self.dino.norm(x)
        x = self.classifier(x)
        return x

model = DinoBinaryClassifier(pretrained_model='dinov2_vits14')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: ", device)
model = model.to(device)
model = model.train()
ciriterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-6)

num_epochs = 15

def train_model(model, train_loader, criterion, optimizer, num_epochs):
     
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            
            outputs = model(images.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        epoch_loss = running_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images.to(device))
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f"Accuracy on the test set: {100 * correct / total: .2f}%") 
    
data_dir = './dataset/plant'       
batch_size = 32
img_size=224
train_loader, test_loader = load_data(data_dir, batch_size, img_size)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: ", device)
model = DinoBinaryClassifier()
model = model.to(device)
ciriterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-6)

num_epochs = 15
train_model(model, train_loader,ciriterion, optimizer, num_epochs)
test_model(model, test_loader)