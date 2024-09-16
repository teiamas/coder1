import torch
import os
import constants
from my_CNN import  LeNet5

def change_extension(filename, new_extension):
    # Split the filename into the base name and the current extension
    base_name, _ = os.path.splitext(filename)
    # Create the new filename with the new extension
    new_filename = f"{base_name}.{new_extension}"
    return new_filename

def pth2onnx(model_file, model, device):
    model.load_state_dict(torch.load(model_file))
    model.eval()
    
    # Create a dummy input tensor with the correct data type
    
    dummy_input = torch.randn(1, 1, constants.VRES, constants.HRES, device=device, dtype=torch.float32)
    
    # Export the model to ONNX
    torch.onnx.export(model, dummy_input, "time_net.onnx", 
                      export_params=True, 
                      opset_version=11, 
                      do_constant_folding=True, 
                      input_names=['input'], 
                      output_names=['minutes', 'seconds', 'deciseconds'],
                      dynamic_axes={'input': {0: 'batch_size'}, 'minutes': {0: 'batch_size'}, 'seconds': {0: 'batch_size'}, 'deciseconds': {0: 'batch_size'}})

if __name__ == '__main__': 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model = TimeNet().to(device) 
    model = LeNet5().to(device) 
    file_name ='time_net.pth' 
    pth2onnx(file_name, model, device)

 