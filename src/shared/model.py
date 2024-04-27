import torch
from tqdm import tqdm
import os
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class ModelManager:
    def __init__(self):
        self.model = self.load_model()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.train_dl = None
        self.val_dl = None

    # ===========================================================

    def load_model(self):

        model = resnet18(weights=ResNet18_Weights.DEFAULT)

        # Free resnet layers
        for param in model.parameters():
            param.requires_grad = False
        
        model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        model.fc = nn.Sequential(nn.Flatten(), nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 1), nn.Sigmoid())
        
        return model
    
    # ===========================================================
    
    def get_model(self):
        return self.model
    
    # ===========================================================       
    
    def get_model_weights(self):
        """
        Retrieves the current weights of the model.

        Returns:
            list of torch.Tensor: A list of tensors representing the weights of the model.
        """
        return [param.data for param in self.model.parameters()]
    
    # ===========================================================

    def set_model_weights(self, weights):
        """
        Sets the model's weights to the given weights, typically used when loading updated weights after federated averaging.

        Params:
            weights (list of torch.Tensor): A list of tensors representing the new weights to be set on the model.
        """
        with torch.no_grad():
            for param, weight in zip(self.model.parameters(), weights):
                param.data.copy_(weight)

    # ===========================================================

    def create_dataloader(self, data_dir, batch_size):
        """Load data from disk and apply transformations."""

        transform = transforms.Compose([
            transforms.Resize((256, 256)),  # Resize images to 256x256
            transforms.ToTensor(),  # Convert images to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images
        ])
        
        # Hello windows ?
        num_workers= 0 if os.name == 'nt' else os.cpu_count()

        dataset = datasets.ImageFolder(root=data_dir, transform=transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        return dataloader
    
    # ===========================================================
    def train_epoch(self, dataloader, optimizer, criterion):

        self.model.train()
        total_loss, total_accuracy = 0, 0

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(self.device), labels.to(self.device).float().view(-1, 1)

            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            predictions = (outputs > 0.5).float()
            total_accuracy += (predictions == labels).float().mean().item()

        return total_loss / len(dataloader), total_accuracy / len(dataloader)

    # ===========================================================
    def validate_epoch(self, dataloader, criterion):

        self.model.eval()

        with torch.no_grad():
            total_loss, total_accuracy = 0, 0

            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device).float().view(-1, 1)

                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
        
                total_loss += loss.item()
                predictions = (outputs > 0.5).float()
                total_accuracy += (predictions == labels).float().mean().item()

            return total_loss / len(dataloader), total_accuracy / len(dataloader)
    # ===========================================================

    def train(self, train_dir, val_dir, num_epochs, learning_rate, batch_size):

        if not self.train_dl:
            self.train_dl = self.create_dataloader(train_dir, batch_size)

        if not self.val_dl:
            self.val_dl = self.create_dataloader(val_dir, batch_size)

        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        for epoch in tqdm(range(num_epochs), desc=f"Training"):
            print(f"Epoch: {epoch+1}/{num_epochs}")
            train_loss, train_acc = self.train_epoch(self.train_dl, optimizer, criterion)
            val_loss, val_acc = self.validate_epoch(self.val_dl, criterion)
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            print(f"====================================")