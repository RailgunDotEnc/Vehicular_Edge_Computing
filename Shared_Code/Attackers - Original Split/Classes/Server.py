import torch, copy
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from collections import defaultdict, Counter
from torch import nn
from settings import LR, NUM_USERS
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

def find_targeted_attack_complex(dict_lHoGs, cosine_dist=False):
    """Construct a set of suspecious of targeted and unreliable clients
    by using [normalized] long HoGs (dict_lHoGs dictionary).
    We use two ways of clustering to find all possible suspicious clients:
      - 1st cluster: Using KMeans (K=2) based on Euclidean distance of
      long_HoGs==> find minority ids.
      - 2nd cluster: Using KMeans (K=2) based on angles between
      long_HoGs to median (that is calculated based on only
      normal clients output from the 1st cluster KMeans).
    """
    id_lHoGs = np.array(list(dict_lHoGs.keys()))
    value_lHoGs = np.array(list(dict_lHoGs.values()))
    cluster_lh1 = KMeans(n_clusters=2, random_state=0).fit(value_lHoGs)
    offset_tAtk_id1 = find_minority_id(cluster_lh1)
    sus_tAtk_id1 = id_lHoGs[list(offset_tAtk_id1)]
    print(f"sus_tAtk_id1: {sus_tAtk_id1}")

    offset_normal_ids = find_majority_id(cluster_lh1)
    normal_ids = id_lHoGs[list(offset_normal_ids)]
    normal_lHoGs = value_lHoGs[list(offset_normal_ids)]
    median_normal_lHoGs = np.median(normal_lHoGs, axis=0)
    d_med_lHoGs = {}
    for idx in id_lHoGs:
        if cosine_dist:
            # cosine similarity between median and all long HoGs points.
            d_med_lHoGs[idx] = np.dot(dict_lHoGs[idx], median_normal_lHoGs)
        else:
            # Euclidean distance
            d_med_lHoGs[idx] = np.linalg.norm(dict_lHoGs[idx]- median_normal_lHoGs)

    cluster_lh2 = KMeans(n_clusters=2, random_state=0).fit(np.array(list(d_med_lHoGs.values())).reshape(-1,1))
    offset_tAtk_id2 = find_minority_id(cluster_lh2)
    sus_tAtk_id2 = id_lHoGs[list(offset_tAtk_id2)]
    print(f"d_med_lHoGs={d_med_lHoGs}")
    print(f"sus_tAtk_id2: {sus_tAtk_id2}")
    sus_tAtk_uRel_id = set(list(sus_tAtk_id1)).union(set(list(sus_tAtk_id2)))
    print(f"sus_tAtk_uRel_id: {sus_tAtk_uRel_id}")
    return sus_tAtk_uRel_id


def find_targeted_attack(dict_lHoGs):
    """Construct a set of suspecious of targeted and unreliable clients
    by using long HoGs (dict_lHoGs dictionary).
      - cluster: Using KMeans (K=2) based on Euclidean distance of
      long_HoGs==> find minority ids.
    """
    id_lHoGs = np.array(list(dict_lHoGs.keys()))
    value_lHoGs = np.array(list(dict_lHoGs.values()))
    cluster_lh1 = KMeans(n_clusters=2, random_state=0).fit(value_lHoGs)
    #cluster_lh = DBSCAN(eps=35, min_samples=7, metric='mahalanobis', n_jobs=-1).fit(value_lHoGs)
    #logging.info(f"DBSCAN labels={cluster_lh.labels_}")
    offset_tAtk_id1 = find_minority_id(cluster_lh1)
    sus_tAtk_id = id_lHoGs[list(offset_tAtk_id1)]
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
        self.all_client_hogs = []
        
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
        self.iter = 0
        self.func = torch.mean
        self.isSaveChanges = False
        self.path_to_aggNet = ""
        self.sims = None
        self.mal_ids = set()
        self.uAtk_ids = set()
        self.tAtk_ids = set()
        self.flip_sign_ids = set()
        self.unreliable_ids = set()
        self.suspicious_id = set()
        self.log_sims = None
        self.log_norms = None
        # At least tao_0 + delay_decision rounds to get first decision.
        self.tao_0 = 3
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
    
    def FedFuncWholeNet(self, clients, deltas, datasize, func):
        '''
        The aggregation rule views the update vectors as stacked vectors (1 by d by n).
        '''
        Delta = copy.deepcopy(self.emptyStates)
        deltas = deltas
        # size is relative to number of samples, actually it is number of batches
        sizes = datasize
        total_s = sum(sizes)
        weights = [s/total_s for s in sizes]
        vecs = [utils.net2vec(delta) for delta in deltas]
        vecs = [vec for vec in vecs if torch.isfinite(vec).all().item()]
        weighted_vecs = [w*v for w,v in zip(weights, vecs)]
        result = func(torch.stack(vecs, 1).unsqueeze(0))  # input as 1 by d by n
        result = result.view(-1)
        utils.vec2net(result, Delta)
        return Delta
    
    def mud_hog(self, clients, deltas, datasize, device):
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
            #sum_hog_i = clients[i].get_sum_hog().detach().cpu().numpy()
            sum_hog_i = self.all_client_hogs[i]
            #L2_sum_hog_i = clients[i].get_L2_sum_hog().detach().cpu().numpy()
            long_HoGs[i] = sum_hog_i

            # shortHoGs
            #sHoG = clients[i].get_avg_grad().detach().cpu().numpy()
            sHoG = get_avg_grad(self.all_client_hogs[i]).cpu().numpy()
            #logging.debug(f"sHoG={sHoG.shape}") # model's total parameters, cifar=sHoG=(11191262,)
            L2_sHoG = np.linalg.norm(sHoG)
            full_norm_short_HoGs.append(sHoG/L2_sHoG)
            short_HoGs[i] = sHoG

            # Exclude the firmed malicious clients
            if i not in self.mal_ids:
                normalized_sHoGs[i] = sHoG/L2_sHoG

        # STAGE 2: Clustering and find malicious clients
        if self.iter >= self.tao_0:
            # STEP 1: Detect FLIP_SIGN gradient attackers
            """By using angle between normalized short HoGs to the median
            of normalized short HoGs among good candidates.
            NOTE: we tested finding flip-sign attack with longHoG, but it failed after long running.
            """
            #flip_sign_id = set()
            """
            median_norm_shortHoG = np.median(np.array([v for v in normalized_sHoGs.values()]), axis=0)
            for i, v in enumerate(full_norm_short_HoGs):
                dot_prod = np.dot(median_norm_shortHoG, v)
                if dot_prod < 0: # angle > 90
                    flip_sign_id.add(i)
                    #logging.debug("Detect FLIP_SIGN client={}".format(i))
            logging.info(f"flip_sign_id={flip_sign_id}")
            """
            """non_mal_sHoGs = dict(short_HoGs) # deep copy dict
            for i in self.mal_ids:
                non_mal_sHoGs.pop(i)
            median_sHoG = np.median(np.array(list(non_mal_sHoGs.values())), axis=0)
            for i, v in short_HoGs.items():
                #logging.info(f"median_sHoG={median_sHoG}, v={v}")
                v = np.array(list(v))
                d_cos = np.dot(median_sHoG, v)/(np.linalg.norm(median_sHoG)*np.linalg.norm(v))
                if d_cos < 0: # angle > 90
                    flip_sign_id.add(i)
                    #logging.debug("Detect FLIP_SIGN client={}".format(i))
            print(f"flip_sign_id={flip_sign_id}")"""


            # STEP 2: Detect UNTARGETED ATTACK
            """ Exclude sign-flipping first, the remaining nodes include
            {NORMAL, ADDITIVE-NOISE, TARGETED and UNRELIABLE}
            we use DBSCAN to cluster them on raw gradients (raw short HoGs),
            the largest cluster is normal clients cluster (C_norm). For the remaining raw gradients,
            compute their Euclidean distance to the centroid (mean or median) of C_norm.
            Then find the bi-partition of these distances, the group of smaller distances correspond to
            unreliable, the other group correspond to additive-noise (Assumption: Additive-noise is fairly
            large (since it is attack) while unreliable's noise is fairly small).
            """

            # Step 2.1: excluding sign-flipping nodes from raw short HoGs:
            """print("===========using shortHoGs for detecting UNTARGETED ATTACK====")
            for i in range(self.num_clients):
                if i in flip_sign_id or i in self.flip_sign_ids:
                    short_HoGs.pop(i)
            id_sHoGs, value_sHoGs = np.array(list(short_HoGs.keys())), np.array(list(short_HoGs.values()))
            # Find eps for MNIST and CIFAR:"""
            """
            dist_1 = {}
            for k,v in short_HoGs.items():
                if k != 1:
                    dist_1[k] = np.linalg.norm(v - short_HoGs[1])
                    logging.info(f"Euclidean distance between 1 and {k} is {dist_1[k]}")

            logging.info(f"Average Euclidean distances between 1 and others {np.mean(list(dist_1.values()))}")
            logging.info(f"Median Euclidean distances between 1 and others {np.median(list(dist_1.values()))}")
            """

            # DBSCAN is mandatory success for this step, KMeans failed.
            # MNIST uses default eps=0.5, min_sample=5
            # CIFAR uses eps=50, min_sample=5 (based on heuristic evaluation Euclidean distance of grad of RestNet18.
            """start_t = time.time()
            cluster_sh = DBSCAN(eps=self.dbscan_eps, n_jobs=-1,
                min_samples=self.dbscan_min_samples).fit(value_sHoGs)
            t_dbscan = time.time() - start_t
            #logging.info(f"CLUSTER DBSCAN shortHoGs took {t_dbscan}[s]")
            # TODO: comment out this line
            #logging.info("labels cluster_sh= {}".format(cluster_sh.labels_))
            offset_normal_ids = find_majority_id(cluster_sh)
            normal_ids = id_sHoGs[list(offset_normal_ids)]
            normal_sHoGs = value_sHoGs[list(offset_normal_ids)]
            normal_cent = np.median(normal_sHoGs, axis=0)
            #logging.debug(f"offset_normal_ids={offset_normal_ids}, normal_ids={normal_ids}")

            # suspicious ids of untargeted attacks and unreliable or targeted attacks.
            offset_uAtk_ids = np.where(cluster_sh.labels_ == -1)[0]
            sus_uAtk_ids = id_sHoGs[list(offset_uAtk_ids)]
            print(f"SUSPECTED UNTARGETED {sus_uAtk_ids}")

            # suspicious_ids consists both additive-noise, targeted and unreliable clients:
            suspicious_ids = [i for i in id_sHoGs if i not in normal_ids] # this includes sus_uAtk_ids
            print(f"suspicious_ids={suspicious_ids}")
            d_normal_sus = {} # distance from centroid of normal to suspicious clients.
            for sid in suspicious_ids:
                d_normal_sus[sid] = np.linalg.norm(short_HoGs[sid]-normal_cent)

            # could not find separate points only based on suspected untargeted attacks.
            #d_sus_uAtk_values = [d_normal_sus[i] for i in sus_uAtk_ids]
            #d_separate = find_separate_point(d_sus_uAtk_values)
            d_separate = find_separate_point(list(d_normal_sus.values()))
            #logging.debug(f"d_normal_sus={d_normal_sus}, d_separate={d_separate}")
            sus_tAtk_uRel_id0, uAtk_id = set(), set()
            for k, v in d_normal_sus.items():
                if v > d_separate and k in sus_uAtk_ids:
                    uAtk_id.add(k)
                else:
                    sus_tAtk_uRel_id0.add(k)
            print(f"This round UNTARGETED={uAtk_id}, sus_tAtk_uRel_id0={sus_tAtk_uRel_id0}")"""


            # STEP 3: Detect TARGETED ATTACK
            """
              - First excluding flip_sign and untargeted attack from.
              - Using KMeans (K=2) based on Euclidean distance of
                long_HoGs==> find minority ids.
            """
            """for i in range(self.num_clients):
                if i in self.flip_sign_ids or i in flip_sign_id:
                    if i in long_HoGs:
                        long_HoGs.pop(i)
                if i in uAtk_id or i in self.uAtk_ids:
                    if i in long_HoGs:
                        long_HoGs.pop(i)"""

            # Using Euclidean distance is as good as cosine distance (which used in MNIST).
            print("=======Using LONG HOGs for detecting TARGETED ATTACK========")
            tAtk_id = find_targeted_attack(long_HoGs)

            # Aggregate, count and record ATTACKERs:
            #self.add_mal_id(flip_sign_id, uAtk_id, tAtk_id)
            self.add_mal_id(tAtk_id)
            print("OVERTIME MALICIOUS client ids ={}".format(self.mal_ids))

            # STEP 4: UNRELIABLE CLIENTS
            """using normalized short HoGs (normalized_sHoGs) to detect unreliable clients
            1st: remove all malicious clients (manipulate directly).
            2nd: find angles between normalized_sHoGs to the median point
            which mostly normal point and represent for aggreation (e.g., Median method).
            3rd: find confident mid-point. Unreliable clients have larger angles
            or smaller cosine similarities.
            """
            """
            for i in self.mal_ids:
                if i in normalized_sHoGs:
                    normalized_sHoGs.pop(i)

            angle_normalized_sHoGs = {}
            # update this value again after excluding malicious clients
            median_norm_shortHoG = np.median(np.array(list(normalized_sHoGs.values())), axis=0)
            for i, v in normalized_sHoGs.items():
                angle_normalized_sHoGs[i] = np.dot(median_norm_shortHoG, v)

            angle_sep_nsH = find_separate_point(list(angle_normalized_sHoGs.values()))
            normal_id, uRel_id = set(), set()
            for k, v in angle_normalized_sHoGs.items():
                if v < angle_sep_nsH: # larger angle, smaller cosine similarity
                    uRel_id.add(k)
                else:
                    normal_id.add(k)
            """
            """for i in self.mal_ids:
                if i in short_HoGs:
                    short_HoGs.pop(i)

            angle_sHoGs = {}
            # update this value again after excluding malicious clients
            median_sHoG = np.median(np.array(list(short_HoGs.values())), axis=0)
            for i, v in short_HoGs.items():
                angle_sHoGs[i] = np.dot(median_sHoG, v)/(np.linalg.norm(median_sHoG)*np.linalg.norm(v))

            angle_sep_sH = find_separate_point(list(angle_sHoGs.values()))
            normal_id, uRel_id = set(), set()
            for k, v in angle_sHoGs.items():
                if v < angle_sep_sH: # larger angle, smaller cosine similarity
                    uRel_id.add(k)
                else:
                    normal_id.add(k)
            print(f"This round UNRELIABLE={uRel_id}, normal_id={normal_id}")
            #logging.debug(f"anlge_normalized_sHoGs={angle_normalized_sHoGs}, angle_sep_nsH={angle_sep_nsH}")
            print(f"anlge_sHoGs={angle_sHoGs}, angle_sep_nsH={angle_sep_sH}")

            for k in range(self.num_clients):
                if k in uRel_id:
                    self.count_unreliable[k] += 1
                    if self.count_unreliable[k] > self.delay_decision:
                        self.unreliable_ids.add(k)
                # do this before decreasing count
                if self.count_unreliable[k] == 0 and k in self.unreliable_ids:
                    self.unreliable_ids.remove(k)
                if k not in uRel_id and self.count_unreliable[k] > 0:
                    self.count_unreliable[k] -= 1
            print("UNRELIABLE clients ={}".format(self.unreliable_ids))"""

            normal_clients = []
            for i, client in enumerate(clients):
                #if i not in self.mal_ids and i not in tAtk_id and i not in uAtk_id:
                if i not in self.mal_ids and i not in tAtk_id:
                    normal_clients.append(client)
            self.normal_clients = normal_clients
        else:
            normal_clients = clients
        out = self.FedFuncWholeNet(normal_clients, deltas, datasize, lambda arr: torch.mean(arr, dim=-1, keepdim=True))
        return out

    
        
        
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
                
                self.idx_copy.clear()
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
        self.all_client_hogs.append(sum_hogs)
        net_glob_server.eval()
      
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
                    out = self.mud_hog(sum_hogs,delta,datasize,device)
                    
                    for i in range(len(self.idx_copy)):
                        if self.idx_copy[i] in out:
                            self.acc_test_collect_user.remove(self.idx_copy[i])
                            self.loss_test_collect_user.remove(self.idx_copy[i])
                            print("Attacker ID {} removed").format(self.idx_copy[i])
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
