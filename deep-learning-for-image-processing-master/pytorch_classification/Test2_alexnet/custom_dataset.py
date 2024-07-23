from torch.utils.data import Dataset, Sampler
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

import glob
import random
import torch
import numpy as np

class CustomSampler(Sampler):
    def __init__(self, data_size, step=1):
        self.data_size = data_size
        self.step = step

    def __iter__(self):
        indices_list = list(range(self.data_size))
        tp = (indices_list[i] for i in range(0, self.data_size, self.step))
        return tp

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
        return image_tensor, torch.tensor(image_label, dtype=torch.int64)  # getitem函数得到的每组数据作为一个数组，B个这样的数组存到batch_list中，将batch_list传给collate_fn函数
    
    def collate_fn(self, batch_list):
        mini_batch_label_list = list()
        mini_batch_img_list = list()
        for img_and_label in batch_list:
            img, label = img_and_label
            mini_batch_img_list.append(img)
            mini_batch_label_list.append(label)
        
        mini_batch_img = torch.stack(mini_batch_img_list, dim=0)
        mini_batch_label = torch.stack(mini_batch_label_list, dim=0)
        
        return mini_batch_img, mini_batch_label  # mini_batch_img.shape = [B, C, h, w](dtype=torch.float32), mini_batch_label.shape = [B](dtype=torch.int64)




if __name__ == "__main__":
    data_root_path = "/home/xsj/dataset/flower_data/flower_photos"
    custom_dataset = CustomDataset(data_root_path)
    data_number = len(custom_dataset)
    sampler = CustomSampler(data_number, step=2)
    data_i = custom_dataset[0]


    train_loader = DataLoader(batch_size=4,
                            #   shuffle=True,
                              dataset=custom_dataset,
                              collate_fn=custom_dataset.collate_fn,
                              sampler=sampler
                              )
    for data in train_loader:
        label, img = data
