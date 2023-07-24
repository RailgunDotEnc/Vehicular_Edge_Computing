import Classes.DatasetManger as DatasetManger

from torch.utils.data import DataLoader
import torch
import time
from collections import deque
from copy import deepcopy
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
        self.ldr_train = DataLoader(DatasetManger.DatasetSplit(dataset_train, idxs), batch_size = 256*4, shuffle = True) #change batch size back to 256*4
        self.ldr_test = DataLoader(DatasetManger.DatasetSplit(dataset_test, idxs_test), batch_size = 256*4, shuffle = True)
        self.layers=layers
        self.is_attacker = False
        self.sum_hog = {param: torch.zeros_like(values) for param, values in net_glob_client.state_dict().items()}
        self.all_client_hogs = []
        self.K_avg = 3 #window size for moving average in MUD-HoG
        self.hog_avg = deque(maxlen = self.K_avg)
        self.init_stateChange(net_glob_client)
        self.originalState = deepcopy(net_glob_client.state_dict())

    
    
    def init_stateChange(self, net_glob_client):
        states = deepcopy(net_glob_client.state_dict())
        for param, values in states.items():
            values *= 0
        self.stateChange = states
        self.avg_delta = deepcopy(states)
        self.sum_hog = deepcopy(states)
        
        
        
    def update(self, net_glob_client):
        newState = net_glob_client.state_dict()
        for p in self.originalState:
            self.stateChange[p] = newState[p] - self.originalState[p]
            self.sum_hog[p] += self.stateChange[p]
            K_ = len(self.hog_avg)
            if K_ == 0:
                self.avg_delta[p] = self.stateChange[p]
            elif K_ < self.K_avg:
                self.avg_delta[p] = (self.avg_delta[p] * K_ + self.stateChange[p]) / (K_ + 1)
            else:
                self.avg_delta[p] += (self.state)
        self.hog_avg.append(self.stateChange)
#copy.deepcopy(net_glob_client).to(device),net_glob_server,device
    def train(self, net_glob_client,net_glob_server,device):
        net_glob_client.train()
        self.all_client_hogs.clear()
        optimizer_client = torch.optim.Adam(net_glob_client.parameters(), lr = self.lr) 
        tempArray=[]
        start_time_local=time.time() 
        #Check new layers==Netclient, new layers == NetServer, server matchers client
        print(self.layers,net_glob_client.Layer_Count,net_glob_server.Layer_Count)
        layer_check_array=[self.layers == net_glob_client.Layer_Count,self.layers == net_glob_server.Layer_Count, net_glob_client.Layer_Count == net_glob_server.Layer_Count]
        if not(layer_check_array[0] and layer_check_array[1] and layer_check_array[2]):
            self.match_netC_netS(net_glob_client,net_glob_server)
        #Run local epochs
        print("TRAINING STARTED")
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
        self.update(net_glob_client)
        sum_hog = self.get_sum_hog()
        self.all_client_hogs.append(sum_hog)
        return net_glob_client.state_dict(), tempArray, net_glob_client.layers, net_glob_client.Layer_Count
    
    def evaluate(self, net_glob_client, ell,net_glob_server,sum_hogs,delta,datasize,device,evaluate=False):
        self.all_client_hogs.clear()
        layer_check_array=[self.layers == net_glob_client.Layer_Count,self.layers == net_glob_server.Layer_Count, net_glob_client.Layer_Count == net_glob_server.Layer_Count]
        print("Check if server, client, and update match: ",layer_check_array[0] and layer_check_array[1] and layer_check_array[2])
        
        if net_glob_client.Layer_Count!=net_glob_server.Layer_Count:
            self.match_netC_netS(net_glob_client,net_glob_server,evaluate)
            
        net_glob_client.eval()
           
        with torch.no_grad():
            len_batch = len(self.ldr_test)
            for batch_idx, (images, labels) in enumerate(self.ldr_test):
                images, labels = images.to(self.device), labels.to(self.device)
                #---------forward prop-------------
                fx,volly  = net_glob_client(images,self.layers)
                client_fx = fx.clone().detach().requires_grad_(True) 
                sum_hog = self.get_sum_hog()
                self.all_client_hogs.append(sum_hog)
                sum_hogs = self.all_client_hogs
                # Sending activations to server 
                self.Global.evaluate_server(client_fx, labels, self.idx, len_batch, ell,net_glob_server,delta,datasize,device,self.layers,volly,sum_hogs)
        return 
    
    def match_netC_netS(self,net_glob_client,net_glob_server,evaluate=False):
        #Check which layers do not match
        layers_C=net_glob_client.Layer_Count
        layers_S=net_glob_server.Layer_Count

        #Second match update, client, and server
        diff3=self.layers[0]-layers_C[0]
        diff4=self.layers[1]-layers_C[1]
        Absdiff=abs(diff3)
        #Client gains layers
        if diff3>0:
            T_array=[]
            for i in range(Absdiff):
                T_array.append(layers_C[0]+Absdiff-i)
            T_array.sort()
            if evaluate==False:
                print(f"Server losses {Absdiff} nodes")
                server_W=net_glob_server.get_weights(net_glob_server.state_dict(), T_array,evaluate)
                net_glob_server.deactivate_layers(T_array)
            elif evaluate==True:
                server_W=net_glob_server.get_weights(net_glob_server.state_dict(), T_array,evaluate)
            print(f"Client gains {Absdiff} nodes")
            net_glob_client.load_state_dict(server_W,strict=False)
            net_glob_client.activate_layers(T_array)
        #Server gains layers
        elif diff4>0:
            T_array=[]
            for i in range(Absdiff):
                T_array.append(layers_C[0]+1-Absdiff+i)
            print(f"Client losses {Absdiff} nodes")
            client_W=net_glob_client.get_weights(net_glob_client.state_dict(), T_array)
            net_glob_client.deactivate_layers(T_array)
            if evaluate==False:
                print(f"Server gains {Absdiff} nodes")
                net_glob_server.load_state_dict(client_W,strict=False)
                net_glob_server.activate_layers(T_array)
    
    
    """def get_weights(self,client_dict,layers):
        layers=[3,4]
        keys=list(client_dict.keys())
        volly={}
        for i in range(len(keys)):
            for j in range(len(layers)):
                if f"layer{layers[j]}." in keys[i]:
                    volly[f"{ keys[i]}"]=client_dict[keys[i]]
        return(volly.keys())"""     
    
    def is_attacking(self):
        return self.is_attacker
    

        
    def get_sum_hog(self):
        return torch.cat(tuple(v.flatten() for v in self.sum_hog.values()))
    
    def get_all_client_hogs(self):
     return torch.stack(self.all_client_hogs)
    
    def getDelta(self):
        return self.stateChange
    
    def get_data_size(self):
        return len(self.ldr_train)
        