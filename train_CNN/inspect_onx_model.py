import onnx

# Load the ONNX model
onnx_model = onnx.load('time_net.onnx')

# Check the model's input and output names
print("Input names:", [input.name for input in onnx_model.graph.input])
print("Output names:", [output.name for output in onnx_model.graph.output])

# Print a human-readable representation of the model
print(onnx.helper.printable_graph(onnx_model.graph))

# Inspect the input tensor
input_tensor = onnx_model.graph.input[0]
input_shape = [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
input_type = input_tensor.type.tensor_type.elem_type

print(f"Input Tensor Name: {input_tensor.name}")
print(f"Input Tensor Shape: {input_shape}")
print(f"Input Tensor Type: {onnx.TensorProto.DataType.Name(input_type)}")

# Inspect the output tensors
for output_tensor in onnx_model.graph.output:
    output_shape = [dim.dim_value for dim in output_tensor.type.tensor_type.shape.dim]
    output_type = output_tensor.type.tensor_type.elem_type
    print(f"Output Tensor Name: {output_tensor.name}")
    print(f"Output Tensor Shape: {output_shape}")
    print(f"Output Tensor Type: {onnx.TensorProto.DataType.Name(output_type)}")
