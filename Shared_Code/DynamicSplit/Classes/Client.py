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
        #Check new layers==Netclient, new layers == NetServer, server matchers client
        print(self.layers,net_glob_client.Layer_Count,net_glob_server.Layer_Count)
        layer_check_array=[self.layers == net_glob_client.Layer_Count,self.layers == net_glob_server.Layer_Count, net_glob_client.Layer_Count == net_glob_server.Layer_Count]
        if not(layer_check_array[0] and layer_check_array[1] and layer_check_array[2]):
            self.match_netC_netS(net_glob_client,net_glob_server)
        #Run local epochs
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
        return net_glob_client.state_dict() , tempArray, net_glob_client.layers,net_glob_client.Layer_Count
    
    def evaluate(self, net_glob_client, ell,net_glob_server,device,evaluate=False):
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
                # Sending activations to server 
                self.Global.evaluate_server(fx, labels, self.idx, len_batch, ell,net_glob_server,device,self.layers,volly)
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
                server_W=net_glob_server.get_weights(net_glob_server.state_dict(), T_array,evaluate)
                net_glob_server.deactivate_layers(T_array)
            elif evaluate==True:
                server_W=net_glob_server.get_weights(net_glob_server.state_dict(), T_array,evaluate)
            net_glob_client.load_state_dict(server_W,strict=False)
            net_glob_client.activate_layers(T_array)
        #Server gains layers
        elif diff4>0:
            T_array=[]
            for i in range(Absdiff):
                T_array.append(layers_C[0]+1-Absdiff+i)
            client_W=net_glob_client.get_weights(net_glob_client.state_dict(), T_array)
            net_glob_client.deactivate_layers(T_array)
            if evaluate==False:
                net_glob_server.load_state_dict(client_W,strict=False)
                net_glob_server.activate_layers(T_array)
        


         