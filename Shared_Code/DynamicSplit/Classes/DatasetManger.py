import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os
from glob import glob 
from sklearn.model_selection import train_test_split
from torchvision import transforms

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label
    

# Custom dataset prepration in Pytorch format
class IMGData(Dataset):
    def __init__(self, df, transform = None):
        self.df = df
        self.transform = transform
        
    def __len__(self):
       
        return len(self.df)
    
    def __getitem__(self, index):
        X = Image.open(self.df['path'][index]).resize((64, 64))
        y = torch.tensor(int(self.df['target'][index]))
        
        if self.transform:
            X = self.transform(X)
        
        return X, y

def dataset_iid(dataset, num_users):
    
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace = False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users  

def SetUpData(num_channels,data_name, img_type):
    df = pd.read_csv(f'data/MyFile({data_name}).csv')



    # merging both folders of dataset -- part1 and part2 -- into a single directory
    imageid_path = {os.path.splitext(os.path.basename(x))[0]: x
                    for x in glob(os.path.join("Data", f'*({data_name})', '*.jpg'))}


    if data_name=="IntelNet":
        for i in range(len(df["image_id"])):
            df["image_id"][i]=str(df["image_id"][i])
    df['path'] = df['image_id'].map(imageid_path.get)
    keys=list(imageid_path.keys())
# =============================================================================
#     for i in range(len(df["image_id"])):
#         for j in range(len(keys)):
#             print("imageid:",df["image_id"][i],"key:",keys[j], "check:",df["image_id"][i]==(keys[j]),type(df["image_id"][i]),type(keys[j]) )
#             if df["image_id"][i]==(keys[j]):
#                 print("Path:",imageid_path[keys[i]])
#                 df["path"][i]=imageid_path[keys[j]]
#                 break
# =============================================================================

    df['cell_type'] = df[' fine_label'].map(img_type.get)
    df['target'] = pd.Categorical(df['cell_type']).codes


    #=============================================================================
    # Train-test split      
    train, test = train_test_split(df, test_size = 0.2)

    train = train.reset_index()
    test = test.reset_index()
    #=============================================================================
    #                         Data preprocessing
    #=============================================================================  
    # Data preprocessing: Transformation 
    mean = [0.485] * num_channels
    std = [0.229] * num_channels

    train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(), 
                            transforms.RandomVerticalFlip(),
                            transforms.Pad(4),
                            transforms.RandomRotation(10),
                            transforms.CenterCrop(64),
                            transforms.ToTensor(), 
                            transforms.Normalize(mean = mean, std = std)
                            ])
        
    test_transforms = transforms.Compose([
                            transforms.Pad(4),
                            transforms.CenterCrop(64),
                            transforms.ToTensor(), 
                            transforms.Normalize(mean = mean, std = std)
                            ])    


    # With augmentation
    dataset_train = IMGData(train, transform = train_transforms)
    dataset_test = IMGData(test, transform = test_transforms)
    return dataset_train, dataset_test