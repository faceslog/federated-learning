from .server import Server
from shared.model import ModelManager

# ========================================
# Global Variables

CHECKPOINT_PATH = ""

# ========================================
model_manager = ModelManager()
server = Server(model_manager, "localhost", 8080)
server.run()