#=============================================================================
# Split learning: ResNet18 on HAM10000
# HAM10000 dataset: Tschandl, P.: The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions (2018), doi:10.7910/DVN/DBW86T

# We have three versions of our implementations
# Version1: without using socket and no DP+PixelDP
# Version2: with using socket but no DP+PixelDP
# Version3: without using socket but with DP+PixelDP

# This program is Version1: Single program simulation 
# ============================================================================
import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import math
import os.path
import pandas as pd

from sklearn.model_selection import train_test_split
from PIL import Image
from glob import glob 
from pandas import DataFrame
import random
import numpy as np
import os
import time as time


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy


SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    print(torch.cuda.get_device_name(0))    

#===================================================================  
from datetime import date, datetime
today = f"{date.today()}".replace("-","_")
timeS=f"{datetime.now().strftime('%H:%M:%S')}".replace(":","_")
program=os.path.basename(__file__)+"_"+today+"_"+timeS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# To print in color -------test/train of the client side
def prRed(skk): print("\033[91m {}\033[00m" .format(skk)) 
def prGreen(skk): print("\033[92m {}\033[00m" .format(skk))     

start_time = time.time()

#===================================================================  
# No. of users
num_users = 2
epochs = 2
frac = 1   # participation of clients; if 1 then 100% clients participate in SL
lr = 0.0001
Tarray = []

#=====================================================================================================
#                           Client-side Model definition
#=====================================================================================================
# Model at client side
class block(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample = None, stride=1):
        super(block, self).__init__()
        self.expansion = 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1,stride=1,padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1,stride=stride,padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride =1, padding=0)
        self.bn3 = nn.BatchNorm2d((out_channels*self.expansion))
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        
    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)


        #x += identity
        x = self.relu(x)
        return x
        
class ResNetClient(nn.Module):
    def __init__(self, block, layers, image_channels):
        super(ResNetClient, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = nn.Sequential (
                nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3, bias = False),
                nn.BatchNorm2d(64),
                nn.ReLU (inplace = True),
                nn.MaxPool2d(kernel_size = 7, stride = 2, padding =1),
            )
        self.layer2 = nn.Sequential  (
                nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1, bias = False),
                nn.BatchNorm2d(64),
                nn.ReLU (inplace = True),
                nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1),
                nn.BatchNorm2d(64),              
            )
        self.layer3 = nn.Sequential (
                nn.Conv2d(64, 64, kernel_size = 3, stride = 2, padding = 1),
                nn.BatchNorm2d(64),
                nn.ReLU (inplace = True),
                nn.Conv2d(64, 64, kernel_size = 3, stride = 2, padding = 1),
                nn.BatchNorm2d(64),       
                )   
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
        
    def forward(self,x):
        #x = self.conv1(x)
        #x = self.bn1(x)
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        #x = self.layer1(x)
        #x = self.layer2(x)

        return x

        
    def _make_layer(self, block, num_residual_blocks, out_channels, stride):
        identity_downsample = None
        layers = []
        
        if stride != 1 or self.in_channels != out_channels:
            identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride = stride), nn.BatchNorm2d(out_channels))
            
        layers.append(block(self.in_channels, out_channels, identity_downsample, stride))
        self.in_channels = out_channels
        
        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, out_channels))
            
        return nn.Sequential(*layers)
    
net_glob_client = ResNetClient(block, [2, 2, 2, 2], image_channels=3)
client_in_channels = net_glob_client.in_channels
if torch.cuda.device_count() > 1:
    print("We use",torch.cuda.device_count(), "GPUs")
    net_glob_client = nn.DataParallel(net_glob_client)   # to use the multiple GPUs; later we can change this to CPUs only 

net_glob_client.to(device)
print(net_glob_client)  


    

class ResNetServer(nn.Module):
    def __init__(self, block, layers, in_channels, num_classes):
        super(ResNetServer, self).__init__()
        self.in_channels = in_channels
        
        
        self.layer4 = self._make_layer(block, layers[0], out_channels = 128, stride = 2)
        self.layer5 = self._make_layer(block, layers[1], out_channels = 256, stride = 2)
        self.layer6 = self._make_layer(block, layers[2], out_channels = 512, stride =2)


        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, num_classes)
        
        
    def forward(self,x):
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

        
    def _make_layer(self, block, num_residual_blocks, out_channels, stride):
        identity_downsample = None
        layers = []
        
        if stride != 1 or self.in_channels != out_channels * 4:
            identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride = stride), nn.BatchNorm2d(out_channels))
            
        layers.append(block(self.in_channels, out_channels, identity_downsample, stride))
        self.in_channels = out_channels
        
        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, out_channels))
            
        return nn.Sequential(*layers)
    
net_glob_server = ResNetServer(block, [2,2,2], client_in_channels, 7) #7 is my numbr of classes
if torch.cuda.device_count() > 1:
    print("We use",torch.cuda.device_count(), "GPUs")
    net_glob_server = nn.DataParallel(net_glob_server)   # to use the multiple GPUs 

net_glob_server.to(device)
print(net_glob_server)      
    
    


#===================================================================================
# For Server Side Loss and Accuracy 
loss_train_collect = []
acc_train_collect = []
loss_test_collect = []
acc_test_collect = []
batch_acc_train = []
batch_loss_train = []
batch_acc_test = []
batch_loss_test = []

criterion = nn.CrossEntropyLoss()
count1 = 0
count2 = 0

#====================================================================================================
#                                  Server Side Program
#====================================================================================================
def calculate_accuracy(fx, y):
    preds = fx.max(1, keepdim=True)[1]
    correct = preds.eq(y.view_as(preds)).sum()
    acc = 100.00 *correct.float()/preds.shape[0]
    return acc

# to print train - test together in each round-- these are made global
acc_avg_all_user_train = 0
loss_avg_all_user_train = 0
loss_train_collect_user = []
acc_train_collect_user = []
loss_test_collect_user = []
acc_test_collect_user = []


#client idx collector
idx_collect = []
l_epoch_check = False
fed_check = False

# Server-side function associated with Training 
def train_server(fx_client, y, l_epoch_count, l_epoch, idx, len_batch):
    global net_glob_server, criterion, device, batch_acc_train, batch_loss_train, l_epoch_check, fed_check
    global loss_train_collect, acc_train_collect, count1, acc_avg_all_user_train, loss_avg_all_user_train, idx_collect
    global loss_train_collect_user, acc_train_collect_user
    
    net_glob_server.train()
    optimizer_server = torch.optim.Adam(net_glob_server.parameters(), lr = lr)

    
    # train and update
    optimizer_server.zero_grad()
    
    fx_client = fx_client.to(device)
    y = y.to(device)
    
    #---------forward prop-------------
    fx_server = net_glob_server(fx_client)
    
    # calculate loss
    loss = criterion(fx_server, y)
    # calculate accuracy
    acc = calculate_accuracy(fx_server, y)
    
    #--------backward prop--------------
    loss.backward()
    dfx_client = fx_client.grad.clone().detach()
    optimizer_server.step()
    
    batch_loss_train.append(loss.item())
    batch_acc_train.append(acc.item())
    
    # server-side model net_glob_server is global so it is updated automatically in each pass to this function
        # count1: to track the completion of the local batch associated with one client
    count1 += 1
    if count1 == len_batch:
        acc_avg_train = sum(batch_acc_train)/len(batch_acc_train)           # it has accuracy for one batch
        loss_avg_train = sum(batch_loss_train)/len(batch_loss_train)
        
        batch_acc_train = []
        batch_loss_train = []
        count1 = 0
        
        prRed('Client{} Train => Local Epoch: {} \tAcc: {:.3f} \tLoss: {:.4f}'.format(idx, l_epoch_count, acc_avg_train, loss_avg_train))
        
                
        # If one local epoch is completed, after this a new client will come
        if l_epoch_count == l_epoch-1:
            
            l_epoch_check = True                # for evaluate_server function - to check local epoch has hitted 
                       
            # we store the last accuracy in the last batch of the epoch and it is not the average of all local epochs
            # this is because we work on the last trained model and its accuracy (not earlier cases)
            
            #print("accuracy = ", acc_avg_train)
            acc_avg_train_all = acc_avg_train
            loss_avg_train_all = loss_avg_train
                        
            # accumulate accuracy and loss for each new user
            loss_train_collect_user.append(loss_avg_train_all)
            acc_train_collect_user.append(acc_avg_train_all)
            
            # collect the id of each new user                        
            if idx not in idx_collect:
                idx_collect.append(idx) 
                #print(idx_collect)
        
        # This is to check if all users are served for one round --------------------
        if len(idx_collect) == num_users:
            fed_check = True                                                  # for evaluate_server function  - to check fed check has hitted
            # all users served for one round ------------------------- output print and update is done in evaluate_server()
            # for nicer display 
                        
            idx_collect = []
            
            acc_avg_all_user_train = sum(acc_train_collect_user)/len(acc_train_collect_user)
            loss_avg_all_user_train = sum(loss_train_collect_user)/len(loss_train_collect_user)
            
            loss_train_collect.append(loss_avg_all_user_train)
            acc_train_collect.append(acc_avg_all_user_train)
            
            acc_train_collect_user = []
            loss_train_collect_user = []
            
    # send gradients to the client               
    return dfx_client

# Server-side functions associated with Testing
def evaluate_server(fx_client, y, idx, len_batch, ell):
    global net_glob_server, criterion, batch_acc_test, batch_loss_test
    global loss_test_collect, acc_test_collect, count2, num_users, acc_avg_train_all, loss_avg_train_all, l_epoch_check, fed_check
    global loss_test_collect_user, acc_test_collect_user, acc_avg_all_user_train, loss_avg_all_user_train
    
    net_glob_server.eval()
  
    with torch.no_grad():
        fx_client = fx_client.to(device)
        y = y.to(device) 
        #---------forward prop-------------
        fx_server = net_glob_server(fx_client)
        
        # calculate loss
        loss = criterion(fx_server, y)
        # calculate accuracy
        acc = calculate_accuracy(fx_server, y)
        
        
        batch_loss_test.append(loss.item())
        batch_acc_test.append(acc.item())
        
               
        count2 += 1
        if count2 == len_batch:
            acc_avg_test = sum(batch_acc_test)/len(batch_acc_test)
            loss_avg_test = sum(batch_loss_test)/len(batch_loss_test)
            
            batch_acc_test = []
            batch_loss_test = []
            count2 = 0
            
            prGreen('Client{} Test =>                   \tAcc: {:.3f} \tLoss: {:.4f}'.format(idx, acc_avg_test, loss_avg_test))
            
            # if a local epoch is completed   
            if l_epoch_check:
                l_epoch_check = False
                
                # Store the last accuracy and loss
                acc_avg_test_all = acc_avg_test
                loss_avg_test_all = loss_avg_test
                        
                loss_test_collect_user.append(loss_avg_test_all)
                acc_test_collect_user.append(acc_avg_test_all)
                
            # if all users are served for one round ----------                    
            if fed_check:
                fed_check = False
                                
                acc_avg_all_user = sum(acc_test_collect_user)/len(acc_test_collect_user)
                loss_avg_all_user = sum(loss_test_collect_user)/len(loss_test_collect_user)
            
                loss_test_collect.append(loss_avg_all_user)
                acc_test_collect.append(acc_avg_all_user)
                acc_test_collect_user = []
                loss_test_collect_user= []
                              
                print("====================== SERVER V1==========================")
                print(' Train: Round {:3d}, Avg Accuracy {:.3f} | Avg Loss {:.3f}'.format(ell, acc_avg_all_user_train, loss_avg_all_user_train))
                print(' Test: Round {:3d}, Avg Accuracy {:.3f} | Avg Loss {:.3f}'.format(ell, acc_avg_all_user, loss_avg_all_user))
                print("==========================================================")
                if len(Tarray) == 0:
                    Tarray.append(time.time()-start_time)
                else:
                    Tarray.append(time.time()-start_time-Tarray[len(Tarray)-1])
         
    return 

#==============================================================================================================
#                                       Clients Side Program
#==============================================================================================================
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

# Client-side functions associated with Training and Testing
class Client(object):
    def __init__(self, net_client_model, idx, lr, device, dataset_train = None, dataset_test = None, idxs = None, idxs_test = None):
        self.idx = idx
        self.device = device
        self.lr = lr
        self.local_ep = 1 
        #self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset_train, idxs), batch_size = 256, shuffle = True)
        self.ldr_test = DataLoader(DatasetSplit(dataset_test, idxs_test), batch_size = 256, shuffle = True)
        

    def train(self, net):
        net.train()
        optimizer_client = torch.optim.Adam(net.parameters(), lr = self.lr) 
        
        for iter in range(self.local_ep):
            len_batch = len(self.ldr_train)
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer_client.zero_grad()
                #---------forward prop-------------
                fx = net(images)
                client_fx = fx.clone().detach().requires_grad_(True)
                
                # Sending activations to server and receiving gradients from server
                dfx = train_server(client_fx, labels, iter, self.local_ep, self.idx, len_batch)
                
                #--------backward prop -------------
                fx.backward(dfx)
                optimizer_client.step()
                            
            
            #prRed('Client{} Train => Epoch: {}'.format(self.idx, ell))
           
        return net.state_dict() 
    
    def evaluate(self, net, ell):
        net.eval()
           
        with torch.no_grad():
            len_batch = len(self.ldr_test)
            for batch_idx, (images, labels) in enumerate(self.ldr_test):
                images, labels = images.to(self.device), labels.to(self.device)
                #---------forward prop-------------
                fx = net(images)
                
                # Sending activations to server 
                evaluate_server(fx, labels, self.idx, len_batch, ell)
            
            #prRed('Client{} Test => Epoch: {}'.format(self.idx, ell))
            
        return          
#=====================================================================================================
# dataset_iid() will create a dictionary to collect the indices of the data samples randomly for each client
# IID HAM10000 datasets will be created based on this
def dataset_iid(dataset, num_users):
    
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace = False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users    
            
#=============================================================================
#                         Data loading 
#============================================================================= 
df = pd.read_csv('data/HAM10000_metadata.csv')
print(df.head())

lesion_type = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}

# merging both folders of HAM1000 dataset -- part1 and part2 -- into a single directory
imageid_path = {os.path.splitext(os.path.basename(x))[0]: x
                for x in glob(os.path.join("data", '*', '*.jpg'))}


#print("path---------------------------------------", imageid_path.get)
df['path'] = df['image_id'].map(imageid_path.get)
df['cell_type'] = df['dx'].map(lesion_type.get)
df['target'] = pd.Categorical(df['cell_type']).codes
print(df['cell_type'].value_counts())
print(df['target'].value_counts())

#==============================================================
# Custom dataset prepration in Pytorch format
class SkinData(Dataset):
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
#=============================================================================
# Train-test split      
train, test = train_test_split(df, test_size = 0.2)

train = train.reset_index()
test = test.reset_index()
#=============================================================================
#                         Data preprocessing
#=============================================================================  
# Data preprocessing: Transformation 
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(), 
                        transforms.RandomVerticalFlip(),
                        transforms.Pad(3),
                        transforms.RandomRotation(10),
                        transforms.CenterCrop(64),
                        transforms.ToTensor(), 
                        transforms.Normalize(mean = mean, std = std)
                        ])
    
test_transforms = transforms.Compose([
                        transforms.Pad(3),
                        transforms.CenterCrop(64),
                        transforms.ToTensor(), 
                        transforms.Normalize(mean = mean, std = std)
                        ])    


# With augmentation
dataset_train = SkinData(train, transform = train_transforms)
dataset_test = SkinData(test, transform = test_transforms)

#----------------------------------------------------------------
dict_users = dataset_iid(dataset_train, num_users)
dict_users_test = dataset_iid(dataset_test, num_users)


#net_glob_client.train()
# this epoch is global epoch, also known as rounds
for iter in range(epochs):
    m = max(int(frac * num_users), 1)
    idxs_users = np.random.choice(range(num_users), m, replace = False)

    # Sequential training/testing among clients      
    for idx in idxs_users:
        local = Client(net_glob_client, idx, lr, device, dataset_train = dataset_train, dataset_test = dataset_test, idxs = dict_users[idx], idxs_test = dict_users_test[idx])
        # Training ------------------
        w_client = local.train(net = copy.deepcopy(net_glob_client).to(device))
              
        # Testing -------------------
        local.evaluate(net = copy.deepcopy(net_glob_client).to(device), ell= iter)
        
        # copy weight to net_glob_client -- use to update the client-side model of the next client to be trained
        net_glob_client.load_state_dict(w_client)
   
#===================================================================================     

print("Training and Evaluation completed!")    

#===============================================================================
# Save output data to .excel file (we use for comparision plots)
round_process = [i for i in range(1, len(acc_train_collect)+1)]
df = DataFrame({'round': round_process,'acc_train':acc_train_collect, 'acc_test':acc_test_collect, 'time':Tarray})     
file_name = program+".xlsx"    
df.to_excel(file_name, sheet_name= "v1_test", index = False)     

#=============================================================================
#                         Program Completed
#============================================================================= 
   