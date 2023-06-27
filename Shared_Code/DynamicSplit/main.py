#=============================================================================
# Split learning: ResNet18
# ============================================================================
#Code imports
from settings import ResNetType, num_users, epochs, local_ep, frac, lr, training_sorce

if training_sorce=="mnist10":
    from Dictionary_Types.dic_mnist10 import data_name, num_channels, img_type
elif training_sorce=="fmnist10":
    from Dictionary_Types.dic_fmnist10 import data_name, num_channels, img_type
elif training_sorce=="cifar10":
    from Dictionary_Types.dic_cifar10 import data_name, num_channels, img_type
elif training_sorce=="cifar100":  
    from Dictionary_Types.dic_cifar100 import data_name, num_channels, img_type
    
from Classes.Baseblock import Baseblock
from Classes.ResNet18SS import ResNet18_server_side
from Classes.ResNet18CS import ResNet18_client_side
from Classes.DatasetManger import dataset_iid, SetUpData
from Classes.Server import Server
from Classes.Client import Client
##################################################################################
import torch
from torch import nn
import os.path
from pandas import DataFrame
import random
import numpy as np
import os
import time
import copy
from datetime import date, datetime

print("#############Setting up#############")
today = f"{date.today()}".replace("-","_")
timeS=f"{datetime.now().strftime('%H:%M:%S')}".replace(":","_")
program=os.path.basename(__file__)+"_"+today+"_"+timeS+data_name
TsArray=[]
TcArray=[]

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    print(torch.cuda.get_device_name(0))    

#===================================================================  
print(f"---------{program}----------")              # this is to identify the program in the slurm outputs files
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)


#=====================================================================================================
#                           Client-side Model definition
#=====================================================================================================
print("#############Assign net_glob_client#############")
net_glob_client = ResNet18_client_side(num_channels)
if torch.cuda.device_count() > 1:
    print("We use",torch.cuda.device_count(), "GPUs")
    net_glob_client = nn.DataParallel(net_glob_client)   # to use the multiple GPUs; later we can change this to CPUs only 

net_glob_client.to(device)
print(net_glob_client)     


#=====================================================================================================
#                           Server-side Model definition
#=====================================================================================================



print("#############Assign net_glob_server#############")
net_glob_server = ResNet18_server_side(Baseblock, ResNetType, len(img_type)) #7 is my numbr of classes
if torch.cuda.device_count() > 1:
    print("We use",torch.cuda.device_count(), "GPUs")
    net_glob_server = nn.DataParallel(net_glob_server)   # to use the multiple GPUs 

net_glob_server.to(device)  





Global=Server()

            
#=============================================================================
#                         Data loading 
#============================================================================= 
print("#############Set up Dataset#############")
dataset_train, dataset_test=SetUpData(num_channels,data_name, img_type)

#----------------------------------------------------------------
dict_users = dataset_iid(dataset_train, num_users)
dict_users_test = dataset_iid(dataset_test, num_users)


# this epoch is global epoch, also known as rounds
print("#############Start AI training#############")
start_time = time.time() 
for iter in range(epochs):
    m = max(int(frac * num_users), 1)
    idxs_users = np.random.choice(range(num_users), m, replace = False)
    tempClientArray=[]
    # Sequential training/testing among clients      
    for idx in idxs_users:
        local = Client(Global,local_ep,net_glob_client, idx, lr, device, dataset_train = dataset_train, dataset_test = dataset_test, idxs = dict_users[idx], idxs_test = dict_users_test[idx])
        # Training ------------------
        w_client,tempArray = local.train(copy.deepcopy(net_glob_client).to(device),net_glob_server,device)
              
        # Testing -------------------
        local.evaluate(copy.deepcopy(net_glob_client).to(device),iter,net_glob_server,device)
        tempClientArray.append(tempArray)
        # copy weight to net_glob_client -- use to update the client-side model of the next client to be trained
        net_glob_client.load_state_dict(w_client)
    TcArray.append(tempClientArray)
    TsArray.append((time.time() - start_time)/60)
   
#===================================================================================     

print("Training and Evaluation completed!")    

#===============================================================================
# Save output data to .excel file (we use for comparision plots)
round_process = [i for i in range(1, len(Global.acc_train_collect)+1)]
df = DataFrame({'round': round_process,'acc_train':Global.acc_train_collect, 'acc_test':Global.acc_test_collect, 'Gobal E Time (m)':TsArray, 'Local e Time per Client (m)': TcArray})     
file_name = f"Results\\{program}_{num_users}.xlsx"
df.to_excel(file_name, sheet_name= "v1_test", index = False)     

#=============================================================================
#                         Program Completed
#============================================================================= 



 












