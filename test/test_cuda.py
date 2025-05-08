import torch

print(
    "CUDA is available: ", torch.cuda.is_available() # Check if CUDA is available
)
print(
    "CUDA device count: ", torch.cuda.device_count() # Get the number of CUDA devices
)
print("Torch version: ", torch.__version__) # Print the PyTorch version
print("CUDA version: ", torch.version.cuda) # Print the CUDA version
