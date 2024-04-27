from .client import Client
from shared.model import ModelManager

model_manager = ModelManager()

client = Client('http://localhost:8080', model_manager)
client.connect_to_server()