# -*- coding: utf-8 -*-

"""
Created on Thu Jul  6 11:13:50 2023

@author: sophi
"""
import Classes.DatasetManger as DatasetManger

from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import time

import Classes.Client as Client



class Attacker_LabelFlipping1to7(Client.Client):
    def __init__(self, global_server, local_ep, layers, net_glob_client, idx, lr, device, dataset_train = None, dataset_test = None, idxs = None, idxs_test = None, source_label=1, target_label=7):
        super(Attacker_LabelFlipping1to7, self).__init__(global_server, local_ep, layers, net_glob_client, idx, lr, device, dataset_train, dataset_test, idxs, idxs_test)
        self.source_label = source_label
        self.target_label = target_label
        self.is_attacking = True
        self.ldr_train = DataLoader(DatasetManger.DatasetSplit(dataset_train, idxs), batch_size = 256*4, shuffle = False)
        #logging.info(f"init ATTACK LABEL Change from {source_label} to {target_label} Client {cid}")
        print(f"init ATTACK LABEL Change from {source_label} to {target_label} Client {idx}")

    """def data_transform(self, data, target):
        #target_ = torch.tensor(list(map(lambda x: self.target_label if x == self.source_label else x, target)))
        #assert target.shape == target_.shape, "Inconsistent target shape"
        #data[0] = data[target-1]
        
        print("***************data transformed***************")
        return data"""
    

    
#copy.deepcopy(net_glob_client).to(device),net_glob_server,device
    def train(self, net_glob_client,net_glob_server,device):
        net_glob_client.train()
        optimizer_client = torch.optim.Adam(net_glob_client.parameters(), lr = self.lr) 
        tempArray=[]
        start_time_local=time.time() 
        #Check new layers==Netclient, new layers == NetServer, server matchers client
        print(self.layers,net_glob_client.Layer_Count,net_glob_server.Layer_Count)
        layer_check_array=[self.layers == net_glob_client.Layer_Count,self.layers == net_glob_server.Layer_Count, net_glob_client.Layer_Count == net_glob_server.Layer_Count]
        if not(layer_check_array[0] and layer_check_array[1] and layer_check_array[2]):
            self.match_netC_netS(net_glob_client,net_glob_server)
        #Run local epochs
        for iter in range(self.local_ep):
            len_batch = len(self.ldr_train)
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                if batch_idx == 0:
                    labels_A = labels.tolist()
                    for i in range(len(labels_A)):
                        if labels_A[i] == 1:
                            labels_A[i] = int(self.target_label)
                            #print("label changed from 1 to ", labels_A[i])
                    labels = torch.tensor(labels_A)  
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer_client.zero_grad()
                #---------forward prop-------------
                #FX holds middle gradient/layer weights
                fx,volly= net_glob_client(images,self.layers)

                client_fx = fx.clone().detach().requires_grad_(True)        
                # Sending activations to server and receiving Y^ from server
                dfx = self.Global.train_server(client_fx, labels, iter, self.local_ep, self.idx, len_batch,net_glob_server,device,self.layers,volly)
                #--------backward prop -------------
                fx.backward(dfx)
                optimizer_client.step()
            tempArray.append((time.time() - start_time_local)/60)
        return net_glob_client.state_dict() , tempArray, net_glob_client.layers,net_glob_client.Layer_Count