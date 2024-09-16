import onnxruntime as ort
import numpy as np
import os
import csv
from PIL import Image
import constants

def preprocess_image(image_path):
    # Load image and convert to grayscale
    image = Image.open(image_path).convert('L')
    # Resize image to 480x640 (height x width)
    #print(f"Before: width:{image.width} x height:{image.height}")
    #image = image.resize((480, 640))
    #print(f"After: width:{image.width} x height:{image.height}")
    # Convert image to numpy array and normalize
    image_array = np.array(image).astype(np.float32) / 255.0
    # Add batch dimension and channel dimension
    image_array = np.expand_dims(image_array, axis=0)
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def run_inference(model_path, input_data):
    # Create ONNX runtime session
    session = ort.InferenceSession(model_path)
    # Get input name for the model
    input_name = session.get_inputs()[0].name
    # Run inference
    outputs = session.run(None, {input_name: input_data})
    return outputs

def format_output(minutes, seconds, deciseconds):
    # Get the index of the maximum value for each output tensor
    m = np.argmax(minutes)
    s = np.argmax(seconds)
    d = np.argmax(deciseconds)
    return f"{m:02}:{s:02}.{d}"

def main(model_path, input_dir, output_csv):
    # Open CSV file for writing
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['file_name', 'mm:ss.d'])
        
        # Iterate over files in the input directory
        for filename in os.listdir(input_dir):
            if filename.endswith(".pgm"):
                file_path = os.path.join(input_dir, filename)
                # Preprocess image
                input_data = preprocess_image(file_path)
                # Run inference
                minutes, seconds, deciseconds = run_inference(model_path, input_data)
                # Format output
                result = format_output(minutes, seconds, deciseconds)
                # Write result to CSV file
                writer.writerow([filename, result])

# Example usage
model_path = 'time_net.onnx'
input_dir = './dataset/frames/00_18_0'
output_csv = 'results.csv'
main(model_path, input_dir, output_csv)
