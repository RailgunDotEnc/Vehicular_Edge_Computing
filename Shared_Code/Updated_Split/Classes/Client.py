import Classes.DatasetManger as DatasetManger

from torch.utils.data import DataLoader
import torch
import time
#==============================================================================================================
#                                       Clients Side Program
#==============================================================================================================

# Client-side functions associated with Training and Testing
class Client(object):
    def __init__(self,global_server,local_ep,layers, net_glob_client, idx, lr, device, dataset_train = None, dataset_test = None, idxs = None, idxs_test = None):
        self.idx = idx
        self.device = device
        self.lr = lr
        self.local_ep = local_ep
        self.Global=global_server
        self.ldr_train = DataLoader(DatasetManger.DatasetSplit(dataset_train, idxs), batch_size = 256*4, shuffle = True)
        self.ldr_test = DataLoader(DatasetManger.DatasetSplit(dataset_test, idxs_test), batch_size = 256*4, shuffle = True)
        self.layers=layers
    
#copy.deepcopy(net_glob_client).to(device),net_glob_server,device
    def train(self, net_glob_client,net_glob_server,device):
        net_glob_client.train()
        optimizer_client = torch.optim.Adam(net_glob_client.parameters(), lr = self.lr) 
        tempArray=[]
        start_time_local=time.time() 
        print("Check if server, client, and update match: ",self.layers == net_glob_client.Layer_Count and net_glob_client.Layer_Count == net_glob_server.Layer_Count and self.layers == net_glob_server.Layer_Count)
        if not(self.layers == net_glob_client.Layer_Count and net_glob_client.Layer_Count == net_glob_server.Layer_Count and self.layers == net_glob_server.Layer_Count):
            self.match_netC_netS(net_glob_client,net_glob_server)
            
        for iter in range(self.local_ep):
            len_batch = len(self.ldr_train)
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
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
        return net_glob_client.state_dict() , tempArray
    
    def evaluate(self, net_glob_client, ell,net_glob_server,device):
        print("Check if server and client match",net_glob_client.Layer_Count==net_glob_server.Layer_Count)
        if net_glob_client.Layer_Count!=net_glob_server.Layer_Count:
            self.match_netC_netS(net_glob_client,net_glob_server)
            
        net_glob_client.eval()
           
        with torch.no_grad():
            len_batch = len(self.ldr_test)
            for batch_idx, (images, labels) in enumerate(self.ldr_test):
                images, labels = images.to(self.device), labels.to(self.device)
                #---------forward prop-------------
                fx,volly  = net_glob_client(images,self.layers)
                # Sending activations to server 
                self.Global.evaluate_server(fx, labels, self.idx, len_batch, ell,net_glob_server,device,self.layers,volly)
        return 
    
    def match_netC_netS(self,selfnet_glob_client,net_glob_server):
        pass
    
    
    def get_weights(self,client_dict,layers):
        layers=[3,4]
        keys=list(client_dict.keys())
        volly={}
        for i in range(len(keys)):
            for j in range(len(layers)):
                if f"layer{layers[j]}." in keys[i]:
                    volly[f"{ keys[i]}"]=client_dict[keys[i]]
        return(volly.keys())     