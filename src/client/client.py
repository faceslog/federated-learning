import socketio
import logging
import torch
import io
import zlib
import hashlib

from typing import Type
from shared.model import ModelManager

class Client:
    def __init__(self: Type, server_url: str, model_manager: ModelManager):
        self.server_url = server_url
        self.sio = socketio.Client()
        self.model_manager = model_manager
        logging.basicConfig(level=logging.INFO)

        # Register events
        self.register_handlers()

    # ===========================================================
    # EVENT HANDLER METHOD
    # ===========================================================
    def register_handlers(self):

        @self.sio.event
        def connect():
            logging.info("Connected to the server")
            self.request_weights()  # Request weights upon connection

        @self.sio.event
        def disconnect():
            logging.info("Disconnected from the server")

        @self.sio.event
        def receive_weights(data):
            try:
                received_checksum = data['checksum']
                weights_data = data['weights']

                if self.compute_checksum(weights_data) == received_checksum:
                    
                    weights_buffer = zlib.decompress(weights_data)
                    weights = torch.load(io.BytesIO(weights_buffer))

                    self.set_weights(weights)
                else:
                    logging.error("Checksum verification failed")

            except Exception as e:
                logging.error(f"Error setting weights: {e}")
    
    # ===========================================================
    # OTHERS CLASS METHODS
    # ===========================================================

    def connect_to_server(self):
        try:
            self.sio.connect(self.server_url)
            self.sio.wait()
        except socketio.exceptions.ConnectionError as e:
            logging.error(f"Connection failed: {e}")

    # ===========================================================

    def compute_checksum(self, data):
        return hashlib.sha256(data).hexdigest()
    
    # ===========================================================

    def request_weights(self):
        """Request model weights from the server."""
        self.sio.emit('request_weights')

    # ===========================================================
    
    def set_weights(self, weights):
        self.model_manager.set_model_weights(weights)
        logging.info("Weights decompressed and set in the model")
