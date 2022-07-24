# Library imports
from cgi import test
import os
import numpy as np
from PIL import Image
import config
import torch
from torchvision import transforms, utils
from itertools import combinations


def get_train_test_images_data_dict(data_path:str, test_size:float):
    """
    Returns two dictionaries for train and validation and other for testing.
    For training and validation, it splits the data as per the **test_size** for each class. 
    :param data_path: Root directory of the image dataset.
    :param test_size: Train test split parameter.
    """
    train_valid_dict = {}
    test_dict = {}
    for class_name in os.listdir(data_path):
        if class_name == '.gitignore':
            continue
        curr_path = os.path.join(data_path, class_name)
        images = []
        for image in os.listdir(curr_path):
            images.append(image)
        indices = np.random.permutation(len(images))
        train_indices = indices[:int(indices.shape[0] * (1-test_size))]
        test_indices = indices[int(indices.shape[0] * (1-test_size)):]

        images = np.array(images)
        train_images_name = images[train_indices]
        test_images_name = images[test_indices]
        train_valid_dict[class_name] = (train_images_name, test_images_name)
    return train_valid_dict, test_dict


class ParasiteDataset(torch.utils.data.dataset.Dataset):
    """
        data_path : path to the folder containing images
        train : to specifiy to load training or testing data 
        transform : Pytorch transforms [required - ToTensor(), optional - rotate, flip]
    """
    def __init__(self, data_path:str, class_dict:dict, train_test_image_data_dict:dict, train:bool=True, test:bool=False, transform:transforms=None):
        
        self.data_path = data_path
        self.train = train
        self.class_dict = class_dict
        self.train_test_image_data_dict = train_test_image_data_dict
        self.data, self.targets = self.load(self.data_path, train, test)
        self.transform = transform
        
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        image = Image.open(self.data[idx])
        
        if self.transform:
            image = self.transform(image)
            
        return image, self.targets[idx]
            

    def load(self, data_path, train, test=False):
        images = []
        targets = []
        for class_name in os.listdir(data_path):
            if class_name == '.gitignore':
                continue
            target = self.class_dict[class_name]
            curr_path = os.path.join(data_path, class_name)
            for image_name in os.listdir(curr_path):
                if not test:
                    if image_name in self.train_test_image_data_dict[class_name][0] and train:
                        images.append(os.path.join(curr_path, image_name))
                        targets.append(target)
                    elif image_name in self.train_test_image_data_dict[class_name][1] and not train:
                        images.append(os.path.join(curr_path, image_name))
                        targets.append(target)
                else:
                    if image_name in self.train_test_image_data_dict[class_name][0] and train:
                        images.append(os.path.join(curr_path, image_name))
                        targets.append(target)
        
        indices = np.random.permutation(len(images))
        images = np.array(images)[indices]
        targets = np.array(targets, dtype=np.int64)[indices]
        return images, targets