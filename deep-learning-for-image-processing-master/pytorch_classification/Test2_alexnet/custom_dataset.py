from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

import glob
import random
import torch
import numpy as np


class CustomDataset(Dataset):
    def __init__(self, data_root_dir):
        super().__init__()
        self.data_root_dir = data_root_dir
        self.label_dict = self.get_label_dict()
        self.data_path_list = self.get_data_path_list()
        self.transform = transforms.Compose([
            transforms.Resize(300),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            # transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.4566, 0.406], std=[0.229, 0.224, 0.225])  # 
        ])
        
    
    def get_label_dict(self):
        data_root_path_list = glob.glob(self.data_root_dir + "/*")
        data_root_path_list.sort()
        label_dict = dict()

        for i, flower_folder_path in enumerate(data_root_path_list):
            flower_name = flower_folder_path.split("/")[-1]
            label_dict[flower_name] = i

        return label_dict

    def get_data_path_list(self):
        data_root_path_list = glob.glob(self.data_root_dir + "/*")
        data_path_list = list()
        for data_root_path in data_root_path_list:
            flower_name = data_root_path.split("/")[-1]
            data_label_id = self.label_dict[flower_name]
            imgs_path = glob.glob(data_root_path + "/*")
            for img_path in imgs_path:
                data_path_list.append({img_path: data_label_id})
        random.shuffle(data_path_list)
        return data_path_list
    
    def __len__(self):
        return len(self.data_path_list)
    
    def __getitem__(self, index):
        image_path, image_label = list(self.data_path_list[index].items())[0]
        image = Image.open(image_path)
        image_tensor = self.transform(image)
        return image_tensor, torch.tensor(image_label, dtype=torch.int64)
    
    def collate_fn(self, batch_list):
        mini_batch_label_list = list()
        mini_batch_img_list = list()
        for img_and_label in batch_list:
            img, label = img_and_label
            mini_batch_img_list.append(img)
            mini_batch_label_list.append(label)
        
        mini_batch_img = torch.stack(mini_batch_img_list, dim=0)
        mini_batch_label = torch.stack(mini_batch_label_list, dim=0)
        
        return mini_batch_img, mini_batch_label




if __name__ == "__main__":
    data_root_path = "/home/xsj/study/deep-learning-for-image-processing/data_set/flower_data/train"
    custom_dataset = CustomDataset(data_root_path)
    data_number = len(custom_dataset)
    data_i = custom_dataset[0]

    train_loader = DataLoader(batch_size=4,
                              shuffle=True,
                              dataset=custom_dataset,
                              collate_fn=custom_dataset.collate_fn,
                              )
    for data in train_loader:
        label, img = data
