import torch
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn
from typing import Type

class ModelManager:
    def __init__(self):
        self.model = self.load_model()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def load_model(self):

        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2) 
        model.add_module('sigmoid', nn.Sigmoid()) 
        
        return model
    
    def get_model(self):
        """Returns the current model."""
        return self.model

    def update_model(self, updates):
        """
        Applies federated averaging to update the model's weights using the updates provided by the clients.

        Params:
            updates (list of torch.Tensor): A list containing the weight tensors from each client's model updates.
        """
        with torch.no_grad():
            # Average each parameter across all client updates
            for param, update in zip(self.model.parameters(), updates):
                param.data = torch.mean(torch.stack([param.data] + update), 0)
    
    def get_model_weights(self):
        """
        Retrieves the current weights of the model.

        Returns:
            list of torch.Tensor: A list of tensors representing the weights of the model.
        """
        return [param.data for param in self.model.parameters()]

    def set_model_weights(self, weights):
        """
        Sets the model's weights to the given weights, typically used when loading updated weights after federated averaging.

        Params:
            weights (list of torch.Tensor): A list of tensors representing the new weights to be set on the model.
        """
        with torch.no_grad():
            for param, weight in zip(self.model.parameters(), weights):
                param.data.copy_(weight)
