import os
import numpy as np
import torch
from torch.utils.data import Dataset
from src.config import CLASSES
class QuickDrawDataset(Dataset):
    def __init__(self, data_path='data', samples_per_class=5000):
        
        self.data_path = data_path
        self.samples_per_class = samples_per_class
        self.num_classes = len(CLASSES)
        self.total_sample = self.num_classes * self.samples_per_class

        self.opened_file = {}
        for cls_name in CLASSES:
            file_path = os.path.join(self.data_path, f'{cls_name}.npy')
            if os.path.exists(file_path):
                self.opened_file[cls_name] = np.load(file_path, mmap_mode='r')


    def __len__(self):
        return self.total_sample
    
    def __getitem__(self, index):

        class_idx = int(index // self.samples_per_class)
        image_idx = int(index % self.samples_per_class)
        
        cls_name = CLASSES[class_idx]
        data = self.opened_file[cls_name]

        image = np.array(data[image_idx], dtype=np.float32)
        image = image / 255.0
        image = image.reshape(1, 28, 28)

        X_tensor = torch.tensor(image, dtype=torch.float32)
        y_tensor = torch.tensor(class_idx, dtype=torch.long)
        return X_tensor, y_tensor
    

if __name__ == '__main__':
    mock_classes = ["banana", "book"]
    if os.path.exists("data"):
        dataset = QuickDrawDataset('data', 5000)
        img, label = dataset[5]
        print(f"Image Shape: {img.shape} | Label: {label}")