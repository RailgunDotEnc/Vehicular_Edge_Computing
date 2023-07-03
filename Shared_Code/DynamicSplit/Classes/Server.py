import torch
from torch import nn
from settings import LR, NUM_USERS

# ====================================================================================================
#                                  Server Side Program
# ====================================================================================================


class Server(object):
    def __init__(self):
        # ===================================================================================
        # For Server Side Loss and Accuracy
        self.loss_train_collect = []
        self.acc_train_collect = []
        self.loss_test_collect = []
        self.acc_test_collect = []
        self.batch_acc_train = []
        self.batch_loss_train = []
        self.batch_acc_test = []
        self.batch_loss_test = []

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

        # client idx collector
        self.idx_collect = []
        self.l_epoch_check = False
        self.fed_check = False

        # ====================================================================================================
        #                                  Server Side Program
        # ====================================================================================================
    def calculate_accuracy(self, fx, y):
        preds = fx.max(1, keepdim=True)[1]
        correct = preds.eq(y.view_as(preds)).sum()
        acc = 100.00 * correct.float()/preds.shape[0]
        return acc

    # Server-side function associated with Training                             
    def train_server(self, fx_client, y, l_epoch_count, l_epoch, idx, len_batch, net_glob_server, device, LayerSplit, volly):

        net_glob_server.train()
        optimizer_server = torch.optim.Adam(
            net_glob_server.parameters(), lr=LR)
        # train and update
        optimizer_server.zero_grad()
        fx_client = fx_client.to(device)
        y = y.to(device)
        # ---------forward prop-------------
        fx_server = net_glob_server(fx_client, LayerSplit, volly)
        # calculate loss
        loss = self.criterion(fx_server, y)
        # calculate accuracy
        acc = self.calculate_accuracy(fx_server, y)
        # --------backward prop--------------
        loss.backward()
        dfx_client = fx_client.grad.clone().detach()
        optimizer_server.step()
        self.batch_loss_train.append(loss.item())
        self.batch_acc_train.append(acc.item())
        # server-side model net_glob_server is global so it is updated automatically in each pass to this function
        # count1: to track the completion of the local batch associated with one client
        self.count1 += 1
        if self.count1 == len_batch:
            # it has accuracy for one batch
            acc_avg_train = sum(self.batch_acc_train)/len(self.batch_acc_train)
            loss_avg_train = sum(self.batch_loss_train) / \
                len(self.batch_loss_train)
            self.batch_acc_train = []
            self.batch_loss_train = []
            self.count1 = 0
            print('Client{} Train => Local Epoch: {} \tAcc: {:.3f} \tLoss: {:.4f}'.format(
                idx, l_epoch_count, acc_avg_train, loss_avg_train))
            # If one local epoch is completed, after this a new client will come
            if l_epoch_count == l_epoch-1:
                # for evaluate_server function - to check local epoch has hitted
                self.l_epoch_check = True
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
                    # print(self.idx_collect)
            # This is to check if all users are served for one round --------------------
            if len(self.idx_collect) == NUM_USERS:
                # for evaluate_server function  - to check fed check has hitted
                self.fed_check = True
                # all users served for one round ------------------------- output print and update is done in evaluate_server()
                # for nicer display

                self.idx_collect = []

                self.acc_avg_all_user_train = sum(
                    self.acc_train_collect_user)/len(self.acc_train_collect_user)
                self.loss_avg_all_user_train = sum(
                    self.loss_train_collect_user)/len(self.loss_train_collect_user)

                self.loss_train_collect.append(self.loss_avg_all_user_train)
                self.acc_train_collect.append(self.acc_avg_all_user_train)

                self.acc_train_collect_user = []
                self.loss_train_collect_user = []

        # send gradients to the client
        return dfx_client

    # Server-side functions associated with Testing
    def evaluate_server(self, fx_client, y, idx, len_batch, ell, net_glob_server, device, LayerSplit, volly):

        net_glob_server.eval()

        with torch.no_grad():
            fx_client = fx_client.to(device)
            y = y.to(device)
            # ---------forward prop-------------
            fx_server = net_glob_server(fx_client, LayerSplit, volly)

            # calculate loss
            loss = self.criterion(fx_server, y)
            # calculate accuracy
            acc = self.calculate_accuracy(fx_server, y)

            self.batch_loss_test.append(loss.item())
            self.batch_acc_test.append(acc.item())

            self.count2 += 1
            if self.count2 == len_batch:
                acc_avg_test = sum(self.batch_acc_test) / \
                    len(self.batch_acc_test)
                loss_avg_test = sum(self.batch_loss_test) / \
                    len(self.batch_loss_test)

                self.batch_acc_test = []
                self.batch_loss_test = []
                self.count2 = 0

                print('Client{} Test =>                   \tAcc: {:.3f} \tLoss: {:.4f}'.format(
                    idx, acc_avg_test, loss_avg_test))

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
                    self.fed_check = False

                    acc_avg_all_user = sum(
                        self.acc_test_collect_user)/len(self.acc_test_collect_user)
                    loss_avg_all_user = sum(
                        self.loss_test_collect_user)/len(self.loss_test_collect_user)

                    self.loss_test_collect.append(loss_avg_all_user)
                    self.acc_test_collect.append(acc_avg_all_user)
                    self.acc_test_collect_user = []
                    self.loss_test_collect_user = []

                    print("====================== SERVER V1==========================")
                    print(' Train: Round {:3d}, Avg Accuracy {:.3f} | Avg Loss {:.3f}'.format(
                        ell, self.acc_avg_all_user_train, self.loss_avg_all_user_train))
                    print(' Test: Round {:3d}, Avg Accuracy {:.3f} | Avg Loss {:.3f}'.format(
                        ell, acc_avg_all_user, loss_avg_all_user))
                    print("==========================================================")

        return
