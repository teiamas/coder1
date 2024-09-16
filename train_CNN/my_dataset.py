import torchvision.transforms as transforms
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch
import os


transform = transforms.Compose([
    #transforms.Resize((480, 640)),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.5], std=[0.5]),
])



class TimeDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.labels = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.labels.iloc[idx, 0])
        image = Image.open(img_name).convert('L')  # Convert to grayscale
        label = self.labels.iloc[idx, 1]
        label = self.time_to_tensor(label)
        if self.transform:
            image = self.transform(image)
        return image, label
    def time_to_tensor(self, time_str):
        m, d_t = time_str.split(':')
        s, d_t = d_t.split('.')
        m, s, d = map(int, [m, s, d_t])
        return torch.tensor([m, s, d], dtype=torch.int32)

