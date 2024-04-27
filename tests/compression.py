import torch
from torchvision.models import resnet18, ResNet18_Weights
import zlib
import io

def main():
    # Load a pretrained ResNet18 model
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    print("Original model loaded with pretrained weights.")
    
    # Simulate saving model weights to a buffer
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    print("Model serialized into buffer.")
    
    # Compress the serialized model
    compressed_weights = zlib.compress(buffer.read())
    print(f"Compressed serialized model: {len(compressed_weights)} bytes")
    
    # Decompress the serialized model
    decompressed_weights = zlib.decompress(compressed_weights)
    print("Decompressed the serialized model.")
    
    # Load weights into a new model instance
    new_model = resnet18(weights=None)  # Initialize without pretrained weights
    new_model.load_state_dict(torch.load(io.BytesIO(decompressed_weights)))
    print("Loaded weights into new model instance.")
    
    # Verify if the weights are loaded correctly by comparing parameters
    for (p1, p2) in zip(model.parameters(), new_model.parameters()):
        if torch.equal(p1.data, p2.data):
            continue
        else:
            print("Mismatch found in parameters, there might be an issue.")
            break
    else:
        print("All parameters match. Serialization, compression, and decompression were successful.")

if __name__ == '__main__':
    main()
