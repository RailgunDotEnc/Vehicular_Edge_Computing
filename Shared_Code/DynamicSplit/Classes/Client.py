import Classes.DatasetManger as DatasetManger

from torch.utils.data import DataLoader
import torch
import time
#==============================================================================================================
#                                       Clients Side Program
#==============================================================================================================

# Client-side functions associated with Training and Testing
class Client(object):
    def __init__(self,global_server,local_ep, net_client_model, idx, lr, device, LayerSplit, dataset_train = None, dataset_test = None, idxs = None, idxs_test = None):
        self.idx = idx
        self.device = device
        self.lr = lr
        self.local_ep = local_ep
        self.Global=global_server
        self.LayerSplit=LayerSplit
        #self.selected_clients = []
        self.ldr_train = DataLoader(DatasetManger.DatasetSplit(dataset_train, idxs), batch_size = 256*4, shuffle = True)
        self.ldr_test = DataLoader(DatasetManger.DatasetSplit(dataset_test, idxs_test), batch_size = 256*4, shuffle = True)
        

    def train(self, net,net_glob_server,device):
        net.train()
        optimizer_client = torch.optim.Adam(net.parameters(), lr = self.lr) 
        tempArray=[]
        start_time_local=time.time() 
        
        for iter in range(self.local_ep):
            len_batch = len(self.ldr_train)
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer_client.zero_grad()
                #---------forward prop-------------
                #FX holds middle gradient/layer weights
                fx,volly= net(images,self.LayerSplit)

                client_fx = fx.clone().detach().requires_grad_(True)            
                # Sending activations to server and receiving Y^ from server
                dfx = self.Global.train_server(client_fx, labels, iter, self.local_ep, self.idx, len_batch,net_glob_server,device,self.LayerSplit,volly)
                #optimizer_client.zero_grad()
                #--------backward prop -------------
                fx.backward(dfx)
                optimizer_client.step()
                            
            
            tempArray.append((time.time() - start_time_local)/60)
        return net.state_dict() , tempArray
    
    def evaluate(self, net, ell,net_glob_server,device):
        net.eval()
           
        with torch.no_grad():
            len_batch = len(self.ldr_test)
            for batch_idx, (images, labels) in enumerate(self.ldr_test):
                images, labels = images.to(self.device), labels.to(self.device)
                #---------forward prop-------------
                fx,volly  = net(images,self.LayerSplit)
                
                # Sending activations to server 
                self.Global.evaluate_server(fx, labels, self.idx, len_batch, ell,net_glob_server,device,self.LayerSplit,volly)
            
            #prRed('Client{} Test => Epoch: {}'.format(self.idx, ell))
            
        return     