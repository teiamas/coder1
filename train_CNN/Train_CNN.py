import torch
import torch.nn as nn
from my_dataset import TimeDataset, transform 
from my_CNN import LeNet5 
from tqdm import tqdm  # Import tqdm for progress bar
import os
import json
import zipfile
import glob
from pth2onnx import pth2onnx
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
import time

def unzip_files(datasets_info, zip_dir):
    for dataset_info in datasets_info:
        base_name = os.path.basename(dataset_info['csv_file']).replace('.csv', '')
        zip_file = os.path.join(zip_dir, f"{base_name}.zip")
        target_unzip_dir = dataset_info['root_dir']
        start_time = time.time() #record start time

        if os.path.exists(target_unzip_dir):
            response = input(f"The directory {target_unzip_dir} already exists. Do you want to delete it and unzip again? (yes/no): ")
            if response.lower() == 'yes':
                os.system(f'rm -rf {target_unzip_dir}')
                print(f"Deleted {target_unzip_dir}")
            else:
                print(f"Skipping {zip_file}")
                continue
        else:
            os.makedirs(target_unzip_dir)
            print(f"Created directory {target_unzip_dir}")

        if os.path.exists(zip_file):
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(target_unzip_dir)
            print(f"Unzipped {zip_file} to {target_unzip_dir}")
        else:
            print(f"{zip_file} does not exist")

def main():
    model_file = 'time_net.pth'
    dataset_dir = './dataset/frames'
    zip_dir = os.path.join(dataset_dir, 'zips')

    # Create datasets.json
    datasets_info = []
    csv_files = glob.glob(os.path.join(dataset_dir, '*.csv'))

    for csv_file in csv_files:
        base_name = os.path.basename(csv_file).replace('.csv', '')
        root_dir = os.path.join(dataset_dir, base_name).replace('\\', '/')
        csv_file = csv_file.replace('\\', '/')
        datasets_info.append({
            "csv_file": csv_file,
            "root_dir": root_dir
        })

    with open('datasets.json', 'w') as f:
        json.dump({"datasets": datasets_info}, f, indent=4)

    # Load datasets from JSON file
    with open('datasets.json', 'r') as f:
        datasets_info = json.load(f)['datasets']
    
    # Unzip files
    unzip_files(datasets_info, zip_dir)
    
    # Training script
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Run on: " + device.type)
    model = LeNet5().to(device)
    if os.path.exists(model_file):
        model.load_state_dict(torch.load(model_file,weights_only=False))
    
    # Define separate loss functions for each output
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    scaler = GradScaler()
    for dataset_info in datasets_info:
        csv_file = dataset_info['csv_file']
        root_dir = dataset_info['root_dir']
        start_time = time.time() #record start time
                
        dataset = TimeDataset(csv_file=csv_file, root_dir=root_dir, transform=transform)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=8, pin_memory=True)
        print("===> "+root_dir)
        for epoch in range(10):  # Number of epochs
            running_loss = 0.0
            for i, (images, labels) in enumerate(tqdm(dataloader, desc=f'Epoch {epoch+1}')):
                images = images.to(device).float()
                labels_mm, labels_ss, labels_d = labels[:, 0].to(device).long(), labels[:, 1].to(device).long(), labels[:, 2].to(device).long()

                optimizer.zero_grad()
                with autocast(device.type):
                    outputs_mm, outputs_ss, outputs_d = model(images)
                    loss_mm = criterion(outputs_mm, labels_mm)
                    loss_ss = criterion(outputs_ss, labels_ss)
                    loss_d = criterion(outputs_d, labels_d)
                    loss = loss_mm + loss_ss + loss_d
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item()
    
            print(f'Epoch {epoch+1}, Loss: {running_loss/len(dataloader)}')
        end_time = time.time()  # Record end time
        elapsed_time = end_time - start_time
        print(f"Elapsed time for {root_dir}: {elapsed_time:.2f} seconds")
        print("<===")
    torch.save(model.state_dict(), model_file) 
    # Convert to ONNX because it is more compatible with TensorFlow we will use on Jetson Nano
    pth2onnx(model_file, model, device)

if __name__ == '__main__':
    main()
