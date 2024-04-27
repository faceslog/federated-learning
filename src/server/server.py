import socketio
import eventlet
import logging
import torch
import io
import zlib
import hashlib

from typing import Type
from shared.model import ModelManager

class Server:
    def __init__(self: Type, model_manager: ModelManager, host: str, port: int):
        self.host = host
        self.port = port
        self.sio = socketio.Server(async_mode='eventlet')
        self.app = socketio.WSGIApp(self.sio)
        self.model_manager = model_manager
        self.clients = set()

        logging.basicConfig(level=logging.INFO)
        self.register_handlers()

    # ===========================================================
    # EVENT HANDLER METHOD
    # ===========================================================
    def register_handlers(self):
        @self.sio.event
        def connect(sid, environ):
            logging.info(f"Client {sid} connected")
            self.clients.add(sid)

        @self.sio.event
        def disconnect(sid):
            logging.info(f"Client {sid} disconnected")
            if sid in self.clients:
                self.clients.remove(sid)

        @self.sio.event
        def request_weights(sid):
            logging.info(f"Client {sid} requested weights")
            self.send_weights(sid)
    # ===========================================================
    # OTHERS CLASS METHODS
    # ===========================================================

    def compute_checksum(self, data):
        return hashlib.sha256(data).hexdigest()

    # ===========================================================
    
    def send_weights(self, sid):
        """Send the model weights to a specific client."""
        try:
            weights = self.model_manager.get_model_weights()

            buffer = io.BytesIO()
            torch.save(weights, buffer)
            buffer.seek(0)
            
            compressed_weights = zlib.compress(buffer.read())
            checksum = self.compute_checksum(compressed_weights)
            self.sio.emit('receive_weights', {'weights': compressed_weights, 'checksum': checksum}, to=sid)

            logging.info(f"Weights compressed and sent to client {sid}")
            
        except Exception as e:
            logging.error(f"Failed to send weights to client {sid}: {e}")

    # ===========================================================

    def run(self):
        """Run the socket.io server"""
        logging.info(f"Running server on {self.host}:{self.port}")
        eventlet.wsgi.server(eventlet.listen((self.host, self.port)), self.app)