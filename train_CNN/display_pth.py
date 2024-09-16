import torch

def display_model_parameters(pth_file):
    # Load the model parameters
    model_params = torch.load(pth_file)
    
    # Iterate through the parameters and print their shapes
    for name, param in model_params.items():
        print(f"Layer: {name}")
        print(f"Shape: {param.shape}")
        print()

# Replace 'your_model.pth' with the path to your .pth file
pth_file = 'time_net.pth' 
display_model_parameters(pth_file)

