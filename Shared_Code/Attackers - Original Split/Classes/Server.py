import torch, copy
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from collections import defaultdict, Counter
from torch import nn
import torch.nn.functional as F
from settings import LR, NUM_USERS, DEFENSES
import time
import Classes.utils.utils as utils

#====================================================================================================
#                                  Server Side Program
#====================================================================================================
def find_separate_point(d):
    # d should be flatten and np or list
    d = sorted(d)
    sep_point = 0
    max_gap = 0
    for i in range(len(d)-1):
        if d[i+1] - d[i] > max_gap:
            max_gap = d[i+1] - d[i]
            sep_point = d[i] + max_gap/2
    return sep_point

def get_avg_grad(vals):
    return torch.cat([v.flatten() for v in vals])

def DBSCAN_cluster_minority(dict_data):
    ids = np.array(list(dict_data.keys()))
    values = np.array(list(dict_data.values()))
    if len(values.shape) == 1:
        values = values.reshape(-1,1)
    cluster_ = DBSCAN(n_jobs=-1).fit(values)
    offset_ids = find_minority_id(cluster_)
    minor_id = ids[list(offset_ids)]
    return minor_id

def Kmean_cluster_minority(dict_data):
    ids = np.array(list(dict_data.keys()))
    values = np.array(list(dict_data.values()))
    if len(values.shape) == 1:
        values = values.reshape(-1,1)
    cluster_ = KMeans(n_clusters=2, random_state=0).fit(values)
    offset_ids = find_minority_id(cluster_)
    minor_id = ids[list(offset_ids)]
    return minor_id

def find_minority_id(clf):
    count_1 = sum(clf.labels_ == 1)
    count_0 = sum(clf.labels_ == 0)
    mal_label = 0 if count_1 > count_0 else 1
    atk_id = np.where(clf.labels_ == mal_label)[0]
    atk_id = set(atk_id.reshape((-1)))
    return atk_id

def find_majority_id(clf):
    counts = Counter(clf.labels_)
    major_label = max(counts, key=counts.get)
    major_id = np.where(clf.labels_ == major_label)[0]
    #major_id = set(major_id.reshape(-1))
    return major_id


def find_targeted_attack(dict_lHoGs):
    """Construct a set of suspecious of targeted and unreliable clients
    by using long HoGs (dict_lHoGs dictionary).
      - cluster: Using KMeans (K=2) based on Euclidean distance of
      long_HoGs==> find minority ids.
    """
    id_lHoGs = np.array(list(dict_lHoGs.keys()))
    value_lHoGs = np.array([tensor.numpy() for tensor_list in dict_lHoGs.values() for tensor in tensor_list])
    print(f"Value_lHoGs shape: {value_lHoGs.shape}")
    cluster_lh1 = KMeans(n_clusters=2, random_state=0).fit(value_lHoGs)
    offset_tAtk_id1 = find_minority_id(cluster_lh1)
    offset_tAtk_id1 = list(offset_tAtk_id1)  # Convert to a list if offset_tAtk_id1 is a set
    for i in range(len(offset_tAtk_id1)):
        offset_tAtk_id1[i] = int(((offset_tAtk_id1[i])/2)-1)
    sus_tAtk_id = id_lHoGs[(offset_tAtk_id1)]  # Convert offset_tAtk_id1 to a list of integers before indexing
    print(f"This round TARGETED ATTACK: {sus_tAtk_id}")
    return sus_tAtk_id



class Server(object):
    def __init__(self):
        #===================================================================================
        # For Server Side Loss and Accuracy 
        self.loss_train_collect = []
        self.acc_train_collect = []
        self.loss_test_collect = []
        self.acc_test_collect = []
        self.batch_acc_train = []
        self.batch_loss_train = []
        self.batch_acc_test = []
        self.batch_loss_test = []
        self.all_client_hogs = [None] * NUM_USERS
        
        self.criterion = nn.CrossEntropyLoss()
        self.count1 = 0
        self.count2 = 0
        
        # to print train - test together in each round-- these are made global
        self.acc_avg_all_user_train = 0
        self.loss_avg_all_user_train = 0
        self.loss_train_collect_user = []
        self.acc_train_collect_user = []
        self.loss_test_collect_user = []
        self.acc_test_collect_user = []


        #client idx collector
        self.idx_collect = []
        self.idx_copy = []
        self.l_epoch_check = False
        self.fed_check = False


        #for mudhog defense
        #self.model = model
        self.emptyStates = None
        #self.init_stateChange()
        self.Delta = None
        self.iter = 1
        self.func = torch.mean
        self.isSaveChanges = False
        self.path_to_aggNet = ""
        self.sims = None
        self.mal_ids = set()
        self.tAtk_ids = set()
        self.log_sims = None
        self.log_norms = None
        # At least tao_0 + delay_decision rounds to get first decision.
        self.tao_0 = 0
        self.delay_decision = 2 # 2 consecutive rounds
        self.pre_mal_id = defaultdict(int)
        self.count_unreliable = defaultdict(int)
        # DBSCAN hyper-parameters:
        self.dbscan_eps = 0.5
        self.dbscan_min_samples=5
    
    def stateChange(self,state_dict):
        states = copy.deepcopy(state_dict)
        for param in states:
            param *= 0
        return states
        
        


        #====================================================================================================
        #                                  Server Side Program
        #====================================================================================================
    def calculate_accuracy(self,fx, y):
        preds = fx.max(1, keepdim=True)[1]
        correct = preds.eq(y.view_as(preds)).sum()
        acc = 100.00 *correct.float()/preds.shape[0]
        return acc
    
    
    def mud_hog(self, clients, deltas, datasize, device,state_dict):
        # long_HoGs for clustering targeted and untargeted attackers
        # and for calculating angle > 90 for flip-sign attack
        long_HoGs = {}

        # normalized_sHoGs for calculating angle > 90 for flip-sign attack
        normalized_sHoGs = {}
        full_norm_short_HoGs = [] # for scan flip-sign each round

        # L2 norm short HoGs are for detecting additive noise,
        # or Gaussian/random noise untargeted attack
        short_HoGs = {}

        # STAGE 1: Collect long and short HoGs.
        for i in range(NUM_USERS):
            # longHoGs
            sum_hog_i = (self.all_client_hogs[i])
            long_HoGs[i] = sum_hog_i

            #sHoG = clients[i].get_avg_grad().detach().cpu().numpy()
            sHoG = get_avg_grad(self.all_client_hogs[i])
            L2_sHoG = np.linalg.norm(sHoG)
            full_norm_short_HoGs.append(sHoG/L2_sHoG)
            short_HoGs[i] = sHoG

          # Exclude the firmed malicious clients
        if i not in self.mal_ids:
            normalized_sHoGs[i] = sHoG/L2_sHoG

        # STAGE 2: Clustering and find malicious clients
        if self.iter >= self.tao_0:
            # Using Euclidean distance is as good as cosine distance (which used in MNIST).
            print("=======Using LONG HOGs for detecting TARGETED ATTACK========")
            tAtk_id = find_targeted_attack(long_HoGs)

            # Aggregate, count and record ATTACKERs:
            #self.add_mal_id(flip_sign_id, uAtk_id, tAtk_id)
            self.add_mal_id(tAtk_id)
            print("OVERTIME MALICIOUS client ids ={}".format(self.mal_ids))

            normal_clients = []
            for i, client in enumerate(clients):
                #if i not in self.mal_ids and i not in tAtk_id and i not in uAtk_id:
                if i not in self.mal_ids and i not in tAtk_id:
                    normal_clients.append(client)
            self.normal_clients = normal_clients
        else:
            normal_clients = clients
        param_float = list(state_dict.keys())
        return

    
        
        
    # Server-side function associated with Training                             #Here
    def train_server(self,fx_client, y, l_epoch_count, l_epoch, idx, len_batch,net_glob_server,device,LayerSplit,volly):
        
        
        net_glob_server.train()
        optimizer_server = torch.optim.Adam(net_glob_server.parameters(), lr = LR)
    
        
        # train and update
        optimizer_server.zero_grad()
        
        fx_client = fx_client.to(device)
        y = y.to(device)
        
        #---------forward prop-------------
        fx_server = net_glob_server(fx_client,LayerSplit,volly)
        
        # calculate loss
        loss = self.criterion(fx_server, y)
        # calculate accuracy
        acc = self.calculate_accuracy(fx_server, y)
        
        #--------backward prop--------------
        loss.backward()
        dfx_client = fx_client.grad.clone().detach()
        optimizer_server.step()
        self.batch_loss_train.append(loss.item())
        self.batch_acc_train.append(acc.item())
        
        # server-side model net_glob_server is global so it is updated automatically in each pass to this function
            # count1: to track the completion of the local batch associated with one client
        self.count1 += 1
        if self.count1 == len_batch:
            acc_avg_train = sum(self.batch_acc_train)/len(self.batch_acc_train)           # it has accuracy for one batch
            loss_avg_train = sum(self.batch_loss_train)/len(self.batch_loss_train)
            
            self.batch_acc_train = []
            self.batch_loss_train = []
            self.count1 = 0
            
            print('Client{} Train => Local Epoch: {} \tAcc: {:.3f} \tLoss: {:.4f}'.format(idx, l_epoch_count, acc_avg_train, loss_avg_train))
            
                    
            # If one local epoch is completed, after this a new client will come
            if l_epoch_count == l_epoch-1:
                
                self.l_epoch_check = True                # for evaluate_server function - to check local epoch has hitted 
                           
                # we store the last accuracy in the last batch of the epoch and it is not the average of all local epochs
                # this is because we work on the last trained model and its accuracy (not earlier cases)
                
                #print("accuracy = ", acc_avg_train)
                acc_avg_train_all = acc_avg_train
                loss_avg_train_all = loss_avg_train
                            
                # accumulate accuracy and loss for each new user
                self.loss_train_collect_user.append(loss_avg_train_all)
                self.acc_train_collect_user.append(acc_avg_train_all)
                
                # collect the id of each new user                        
                if idx not in self.idx_collect:
                    self.idx_collect.append(idx) 
                    #print(self.idx_collect)
            
            # This is to check if all users are served for one round --------------------
            if len(self.idx_collect) == NUM_USERS:
                self.fed_check = True                                                  # for evaluate_server function  - to check fed check has hitted
                # all users served for one round ------------------------- output print and update is done in evaluate_server()
                # for nicer display 
                
                self.idx_copy = []
                self.idx_copy = copy.deepcopy(self.idx_collect)
                self.idx_collect = []
                
                self.acc_avg_all_user_train = sum(self.acc_train_collect_user)/len(self.acc_train_collect_user)
                self.loss_avg_all_user_train = sum(self.loss_train_collect_user)/len(self.loss_train_collect_user)
                
                self.loss_train_collect.append(self.loss_avg_all_user_train)
                self.acc_train_collect.append(self.acc_avg_all_user_train)
                
                self.acc_train_collect_user = []
                self.loss_train_collect_user = []
                
        # send gradients to the client               
        return dfx_client
        
    # Server-side functions associated with Testing
    def evaluate_server(self,fx_client, y, idx, len_batch, ell,net_glob_server,delta,datasize,device,LayerSplit,volly,sum_hogs):
        
        """print(type(device))
        print(device)
        device = torch.device(device)"""
        #fx_client_l = fx_client.tolist()
        # Create a dictionary to store tensors with unique shapes
        # Create a list to store the lengths of tensors along the first dimension
        self.all_client_hogs[idx] = (sum_hogs)
        

      
        with torch.no_grad():
            fx_client = fx_client.to(device)
            y = y.to(device) 
            #---------forward prop-------------
            fx_server = net_glob_server(fx_client,LayerSplit,volly)
            
            # calculate loss
            loss = self.criterion(fx_server, y)
            # calculate accuracy
            acc = self.calculate_accuracy(fx_server, y)
            
            
            self.batch_loss_test.append(loss.item())
            self.batch_acc_test.append(acc.item())
            
                   
            self.count2 += 1
            if self.count2 == len_batch:
                acc_avg_test = sum(self.batch_acc_test)/len(self.batch_acc_test)
                loss_avg_test = sum(self.batch_loss_test)/len(self.batch_loss_test)
                
                self.batch_acc_test = []
                self.batch_loss_test = []
                self.count2 = 0
                
                print('Client{} Test =>                   \tAcc: {:.3f} \tLoss: {:.4f}'.format(idx, acc_avg_test, loss_avg_test))
                
                # if a local epoch is completed   
                if self.l_epoch_check:
                    self.l_epoch_check = False
                    
                    # Store the last accuracy and loss
                    acc_avg_test_all = acc_avg_test
                    loss_avg_test_all = loss_avg_test
                            
                    self.loss_test_collect_user.append(loss_avg_test_all)
                    self.acc_test_collect_user.append(acc_avg_test_all)
                    
                # if all users are served for one round ----------                    
                if self.fed_check:
                    print("At fed check")
                    self.fed_check = False
                    state_dict = net_glob_server.state_dict()
                    changedStates = self.stateChange(state_dict)
                    self.emptyStates = changedStates
                    if DEFENSES:
                        self.mud_hog(sum_hogs,delta,datasize,device,state_dict)
                        
                        
                        for i in range(NUM_USERS):
                            for l_tuple in self.mal_ids:
                                if i in l_tuple:
                                    self.acc_test_collect_user.remove(self.acc_test_collect_user[i])
                                    self.loss_test_collect_user.remove(self.loss_test_collect_user[i])
                                    print("Attacker removed")
                        print("Attacker check complete")
                                    
                    acc_avg_all_user = sum(self.acc_test_collect_user)/len(self.acc_test_collect_user)
                    loss_avg_all_user = sum(self.loss_test_collect_user)/len(self.loss_test_collect_user)
                
                    self.loss_test_collect.append(loss_avg_all_user)
                    self.acc_test_collect.append(acc_avg_all_user)
                    self.acc_test_collect_user = []
                    self.loss_test_collect_user= []
                                  
                    print("====================== SERVER V1==========================")
                    print(' Train: Round {:3d}, Avg Accuracy {:.3f} | Avg Loss {:.3f}'.format(ell, self.acc_avg_all_user_train, self.loss_avg_all_user_train))
                    print(' Test: Round {:3d}, Avg Accuracy {:.3f} | Avg Loss {:.3f}'.format(ell, acc_avg_all_user, loss_avg_all_user))
                    print("==========================================================")
             
        return 
    
    def get_weights(self,server_dict,layers):
        layers=[5,6]
        keys=list(server_dict.keys())
        volly={}
        for i in range(len(keys)):
            for j in range(len(layers)):
                if f"layer{layers[j]}." in keys[i]:
                    volly[f"{ keys[i]}"]=server_dict[keys[i]]
        print(volly.keys())
        
    def add_mal_id(self, mal_id):
        self.mal_ids.add(tuple(mal_id))
        return