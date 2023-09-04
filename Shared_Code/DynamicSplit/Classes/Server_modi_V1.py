import torch
from torch import nn
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize

import numpy as np
from settings import LR, NUM_USERS
from itertools import cycle

from settings import LR, NUM_USERS
     
class Server(object):
    def __init__(self):
        
        # For Server Side Loss and Accuracy 
        self.loss_train_collect = []
        self.acc_train_collect = []
        self.precision_train_collect = []
        self.recall_train_collect = []
        self.f1_train_collect = []
        self.gmean_micro_train_collect = []
        self.gmean_macro_train_collect = []
        
        self.loss_test_collect = []
        self.acc_test_collect = []
        self.precision_test_collect = []
        self.recall_test_collect = []
        self.f1_test_collect = []
        self.gmean_micro_test_collect = []
        self.gmean_macro_test_collect = []
        
        self.batch_acc_train = []
        self.batch_loss_train = []
        self.batch_precision_train = []
        self.batch_recall_train = []
        self.batch_f1_train = []
        self.batch_gmean_micro_train = []
        self.batch_gmean_macro_train = []
        
        self.batch_acc_test = []
        self.batch_loss_test = []
        self.batch_precision_test = []
        self.batch_recall_test = []
        self.batch_f1_test = []
        self.batch_gmean_micro_test = []
        self.batch_gmean_macro_test = []
        
        # To store ROC curve data
        self.true_labels = []
        self.pred_probs = []
        
        self.criterion = nn.CrossEntropyLoss()
        self.count1 = 0
        self.count2 = 0
        
        # to print train - test together in each round-- these are made global
        self.acc_avg_all_user_train = 0
        self.loss_avg_all_user_train = 0
        self.precision_avg_all_user_train = 0
        self.recall_avg_all_user_train = 0
        self.f1_avg_all_user_train = 0
        self.gmean_micro_avg_all_user_train = 0
        self.gmean_macro_avg_all_user_train = 0
        
        self.loss_train_collect_user = []
        self.acc_train_collect_user = []
        self.precision_train_collect_user = []
        self.recall_train_collect_user = []
        self.f1_train_collect_user = []
        self.gmean_macro_train_collect_user = []
        self.gmean_micro_train_collect_user = []
        
        self.loss_test_collect_user = []
        self.acc_test_collect_user = []
        self.precision_test_collect_user = []
        self.recall_test_collect_user = []
        self.f1_test_collect_user = []
        self.gmean_micro_test_collect_user = []
        self.gmean_macro_test_collect_user = []

        #client idx collector
        self.idx_collect = []
        self.l_epoch_check = False
        self.fed_check = False
        
        self.tick=0    
            
    # Inside Server class

    def get_true_labels(self):
        return self.true_labels

    def get_pred_probs(self):
        return self.pred_probs
    
    def calculate_accuracy(self, fx, y):
        preds = fx.max(1, keepdim=True)[1]
        correct = preds.eq(y.view_as(preds)).sum()
        acc = 100.00 * correct.float() / preds.shape[0]
        return acc, preds
    
    def calculate_metrics(self, fx, y):
        preds = fx.max(1, keepdim=True)[1].cpu().numpy()
        y_true = y.cpu().numpy()

        precision = precision_score(y_true, preds, average='macro',zero_division=1)
        recall = recall_score(y_true, preds, average='macro')
        f1 = f1_score(y_true, preds, average='macro')

        return precision, recall, f1
    
    def calculate_gmean(self, y_true, y_pred, average_scheme='macro'):
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):  # Binary classification
            tn, fp, fn, tp = cm.ravel()
            sensitivity = tp / (tp + fn)
            specificity = tn / (tn + fp)
        else:  # Multi-class classification
            sensitivity = recall_score(y_true, y_pred, average=average_scheme)
            row_sums = cm.sum(axis=1)
            specificity_list = []
            for i in range(cm.shape[0]):
                true_negative = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
                specificity_list.append(true_negative / (true_negative + cm[:, i].sum() - cm[i, i]))
            if average_scheme == 'macro':
                specificity = np.mean(specificity_list)
            elif average_scheme == 'micro':
                specificity = sum(specificity_list) / sum(row_sums)
        return np.sqrt(sensitivity * specificity)
        
    # Server-side function associated with Training                             #Here
    def train_server(self,fx_client, y, l_epoch_count, l_epoch, idx, len_batch, net_glob_server, device, LayerSplit, volly):
        
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
        
        # Calculate all metrics
        acc, preds = self.calculate_accuracy(fx_server, y)
        precision, recall, f1 = self.calculate_metrics(fx_server, y)
        gmean_macro = self.calculate_gmean(y.cpu().numpy(), preds.cpu().numpy(), 'macro')
        gmean_micro = self.calculate_gmean(y.cpu().numpy(), preds.cpu().numpy(), 'micro')
        
        #--------backward prop--------------
        loss.backward()
        dfx_client = fx_client.grad.clone().detach()
        optimizer_server.step()
        
        self.batch_loss_train.append(loss.item())
        self.batch_acc_train.append(acc.item())
        self.batch_precision_train.append(precision)
        self.batch_recall_train.append(recall)
        self.batch_f1_train.append(f1)
        self.batch_gmean_macro_train.append(gmean_macro)
        self.batch_gmean_micro_train.append(gmean_micro)
                
        # server-side model net_glob_server is global so it is updated automatically in each pass to this function
            # count1: to track the completion of the local batch associated with one client
        self.count1 += 1
        if self.count1 == len_batch:
            acc_avg_train = sum(self.batch_acc_train)/len(self.batch_acc_train)           # it has accuracy for one batch
            loss_avg_train = sum(self.batch_loss_train)/len(self.batch_loss_train)
            precision_avg_train = sum(self.batch_precision_train) / len(self.batch_precision_train)
            recall_avg_train = sum(self.batch_recall_train) / len(self.batch_recall_train)
            f1_avg_train = sum(self.batch_f1_train) / len(self.batch_f1_train)
            gmean_macro_avg_train = sum(self.batch_gmean_macro_train) / len(self.batch_gmean_macro_train)
            gmean_micro_avg_train = sum(self.batch_gmean_micro_train) / len(self.batch_gmean_micro_train)

            self.batch_acc_train = []
            self.batch_loss_train = []
            self.batch_precision_train = []
            self.batch_recall_train = []
            self.batch_f1_train = []
            self.batch_gmean_macro_train = []
            self.batch_gmean_micro_train = []
            
            self.count1 = 0
            
            print('Client{} Train => Local Epoch: {} \tAcc: {:.3f} \tLoss: {:.4f} \tPrecision: {:.3f} \tRecall: {:.3f} \tF1-score: {:.3f} \tG-mean (Macro): {:.3f} \tG-mean (Micro): {:.3f}'.format(idx, l_epoch_count, acc_avg_train, loss_avg_train, precision_avg_train, recall_avg_train, f1_avg_train, gmean_macro_avg_train, gmean_micro_avg_train))
            
            # If one local epoch is completed, after this a new client will come
            if l_epoch_count == l_epoch-1:
                
                self.l_epoch_check = True # to evaluate_server function-to check local epoch has hitted 
                           
# we store the last accuracy in the last batch of the epoch and it is not the average of all local epochs
# this is because we work on the last trained model and its accuracy (not earlier cases)
                
                #print("accuracy = ", acc_avg_train)
                acc_avg_train_all = acc_avg_train
                loss_avg_train_all = loss_avg_train
                precision_avg_train_all = precision_avg_train
                recall_avg_train_all = recall_avg_train
                f1_avg_train_all = f1_avg_train
                gmean_micro_avg_train_all = gmean_micro_avg_train
                gmean_macro_avg_train_all = gmean_macro_avg_train
                            
                # accumulate accuracy and loss for each new user
                self.loss_train_collect_user.append(loss_avg_train_all)
                self.acc_train_collect_user.append(acc_avg_train_all)
                self.precision_train_collect_user.append(precision_avg_train_all)
                self.recall_train_collect_user.append(recall_avg_train_all)
                self.f1_train_collect_user.append(f1_avg_train_all)
                self.gmean_macro_train_collect_user.append(gmean_macro_avg_train_all)
                self.gmean_micro_train_collect_user.append(gmean_micro_avg_train_all)
                
                # collect the id of each new user                        
                if idx not in self.idx_collect:
                    self.idx_collect.append(idx) 
                    #print(self.idx_collect)
            
            # This is to check if all users are served for one round --------------------
            if len(self.idx_collect) == NUM_USERS:
                self.fed_check = True                                                  # for evaluate_server function  - to check fed check has hitted
                # all users served for one round ------------------------- output print and update is done in evaluate_server()
                # for nicer display 
                            
                self.idx_collect = []
                
                self.acc_avg_all_user_train = sum(self.acc_train_collect_user)/len(self.acc_train_collect_user)
                self.loss_avg_all_user_train = sum(self.loss_train_collect_user)/len(self.loss_train_collect_user)
                self.precision_avg_all_user_train = sum(self.precision_train_collect_user)/len(self.precision_train_collect_user)
                self.recall_avg_all_user_train = sum(self.recall_train_collect_user)/len(self.recall_train_collect_user)
                self.f1_avg_all_user_train = sum(self.f1_train_collect_user)/len(self.f1_train_collect_user)
                self.gmean_micro_avg_all_user_train = sum(self.gmean_micro_train_collect_user)/len(self.gmean_micro_train_collect_user)
                self.gmean_macro_avg_all_user_train = sum(self.gmean_macro_train_collect_user)/len(self.gmean_macro_train_collect_user)
                
                self.loss_train_collect.append(self.loss_avg_all_user_train)
                self.acc_train_collect.append(self.acc_avg_all_user_train)
                self.precision_train_collect.append(self.precision_avg_all_user_train)
                self.recall_train_collect.append(self.recall_avg_all_user_train)
                self.f1_train_collect.append(self.f1_avg_all_user_train)
                self.gmean_micro_train_collect.append(self.gmean_micro_avg_all_user_train)
                self.gmean_macro_train_collect.append(self.gmean_macro_avg_all_user_train)
                
                self.acc_train_collect_user = []
                self.loss_train_collect_user = []
                self.precision_train_collect_user = []
                self.recall_train_collect_user = []
                self.f1_train_collect_user = []
                self.gmean_micro_train_collect_user = []
                self.gmean_macro_train_collect_user = []
                
        # send gradients to the client               
        return dfx_client
       
    # Server-side functions associated with Testing
    def evaluate_server(self,fx_client, y, idx, len_batch, ell, net_glob_server, device, LayerSplit, volly):
        
        net_glob_server.eval()
      
        with torch.no_grad():
            fx_client = fx_client.to(device)
            y = y.to(device) 
            #---------forward prop-------------
            fx_server = net_glob_server(fx_client,LayerSplit,volly)
            
            # calculate loss
            loss = self.criterion(fx_server, y)
            
            # Save true labels and predicted probabilities
            self.true_labels.extend(y.cpu().numpy())
            self.pred_probs.extend(fx_server.cpu().detach().numpy())
        
            # Calculate accuracy and get predicted labels
            acc, preds = self.calculate_accuracy(fx_server, y)
            precision, recall, f1 = self.calculate_metrics(fx_server, y)
            gmean_macro = self.calculate_gmean(y.cpu().numpy(), preds.cpu().numpy(), 'macro')
            gmean_micro = self.calculate_gmean(y.cpu().numpy(), preds.cpu().numpy(), 'micro')   
            self.batch_acc_test.append(acc.item())
            self.batch_loss_test.append(loss.item())
            self.batch_precision_test.append(precision)
            self.batch_recall_test.append(recall)
            self.batch_f1_test.append(f1)            
            self.batch_gmean_macro_test.append(gmean_macro)
            self.batch_gmean_micro_test.append(gmean_micro)            
                   
            self.count2 += 1
            if self.count2 == len_batch:
                acc_avg_test = sum(self.batch_acc_test)/len(self.batch_acc_test)
                loss_avg_test = sum(self.batch_loss_test)/len(self.batch_loss_test)
                precision_avg_test = sum(self.batch_precision_test)/len(self.batch_precision_test)
                recall_avg_test = sum(self.batch_recall_test)/len(self.batch_recall_test)
                f1_avg_test = sum(self.batch_f1_test)/len(self.batch_f1_test)
                gmean_micro_avg_test = sum(self.batch_gmean_micro_test)/len(self.batch_gmean_micro_test)
                gmean_macro_avg_test = sum(self.batch_gmean_macro_test)/len(self.batch_gmean_macro_test)
                
                self.batch_acc_test = []
                self.batch_loss_test = []
                self.batch_precision_test = []
                self.batch_recall_test = []
                self.batch_f1_test = []
                self.batch_gmean_micro_test = []
                self.batch_gmean_macro_test = []
                
                self.count2 = 0
                
                print('Client{} Test => \tAcc: {:.3f} \tLoss: {:.4f} \tPrecision: {:.3f} \tRecall: {:.3f} \tF1-score: {:.3f} \tG-mean (Micro): {:.3f} \tG-mean (Macro): {:.3f}'.format(idx, acc_avg_test, loss_avg_test, precision_avg_test, recall_avg_test, f1_avg_test, gmean_micro_avg_test, gmean_macro_avg_test))
                
                # if a local epoch is completed   
                if self.l_epoch_check:
                    self.l_epoch_check = False
                    
                    # Store the last accuracy and loss
                    acc_avg_test_all = acc_avg_test
                    loss_avg_test_all = loss_avg_test
                    precision_avg_test_all = precision_avg_test
                    recall_avg_test_all = recall_avg_test
                    f1_avg_test_all = f1_avg_test
                    gmean_macro_avg_test_all = gmean_macro_avg_test
                    gmean_micro_avg_test_all = gmean_micro_avg_test
                    
                    self.loss_test_collect_user.append(loss_avg_test_all)
                    self.acc_test_collect_user.append(acc_avg_test_all)
                    self.precision_test_collect_user.append(precision_avg_test_all)
                    self.recall_test_collect_user.append(recall_avg_test_all)
                    self.f1_test_collect_user.append(f1_avg_test_all)
                    self.gmean_micro_test_collect_user.append(gmean_micro_avg_test_all)
                    self.gmean_macro_test_collect_user.append(gmean_macro_avg_test_all)
                    
                # if all users are served for one round ----------                    
                if self.fed_check:
                    self.fed_check = False
                                    
                    acc_avg_all_user = sum(self.acc_test_collect_user)/len(self.acc_test_collect_user)
                    loss_avg_all_user = sum(self.loss_test_collect_user)/len(self.loss_test_collect_user)
                    precision_avg_all_user = sum(self.precision_test_collect_user)/len(self.precision_test_collect_user)
                    recall_avg_all_user = sum(self.recall_test_collect_user)/len(self.recall_test_collect_user)
                    f1_avg_all_user = sum(self.f1_test_collect_user)/len(self.f1_test_collect_user)
                    gmean_micro_avg_all_user = sum(self.gmean_micro_test_collect_user)/len(self.gmean_micro_test_collect_user)
                    gmean_macro_avg_all_user = sum(self.gmean_macro_test_collect_user)/len(self.gmean_macro_test_collect_user)
                
                    self.loss_test_collect.append(loss_avg_all_user)
                    self.acc_test_collect.append(acc_avg_all_user)
                    self.precision_test_collect.append(precision_avg_all_user)
                    self.recall_test_collect.append(recall_avg_all_user)
                    self.f1_test_collect.append(f1_avg_all_user)
                    self.gmean_micro_test_collect.append(gmean_micro_avg_all_user)
                    self.gmean_macro_test_collect.append(gmean_macro_avg_all_user)
                    
                    self.acc_test_collect_user = []
                    self.loss_test_collect_user= []
                    self.precision_test_collect_user = []
                    self.recall_test_collect_user= []
                    self.f1_test_collect_user = []
                    self.gmean_micro_test_collect_user= []
                    self.gmean_macro_test_collect_user = []
                                  
                    print("====================== SERVER V1==========================")
                    print(' Train: Round {:3d}, Avg Accuracy {:.3f} | Avg Loss {:.3f} | Avg Precision {:.3f} | Avg Recall {:.3f} | Avg F1 {:.3f} | Avg G-mean (micro) {:.3f} | Avg G-mean (macro) {:.3f}'.format(ell, self.acc_avg_all_user_train, self.loss_avg_all_user_train, self.precision_avg_all_user_train, self.recall_avg_all_user_train, self.f1_avg_all_user_train, self.gmean_micro_avg_all_user_train, self.gmean_macro_avg_all_user_train))
                    
                    print(' Test: Round {:3d}, Avg Accuracy {:.3f} | Avg Loss {:.3f} | Avg Precision {:.3f} | Avg Recall {:.3f} | Avg F1 {:.3f} | Avg G-mean (micro) {:.3f} | Avg G-mean (macro) {:.3f}'.format(ell, acc_avg_all_user, loss_avg_all_user, precision_avg_all_user, recall_avg_all_user, f1_avg_all_user, gmean_micro_avg_all_user, gmean_macro_avg_all_user))
                    print("==========================================================")
                    
#                     print('True Labels', self.true_labels[:10])
#                     print('Predictions', self.pred_probs[:10])
             
        return 
    
    
