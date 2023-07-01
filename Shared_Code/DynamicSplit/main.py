#=============================================================================
# Split learning: ResNet18
# ============================================================================
#Code imports
from settings import RESNETTYPE, NUM_USERS, EPOCHS, LOCAL_EP, FRAC, LR, TRAINING_SORCE

if TRAINING_SORCE=="mnist10":
    from Dictionary_Types.dic_mnist10 import DATA_NAME, NUM_CHANNELS, IMG_TYPE
elif TRAINING_SORCE=="fmnist10":
    from Dictionary_Types.dic_fmnist10 import DATA_NAME, NUM_CHANNELS, IMG_TYPE
elif TRAINING_SORCE=="cifar10":
    from Dictionary_Types.dic_cifar10 import DATA_NAME, NUM_CHANNELS, IMG_TYPE
elif TRAINING_SORCE=="cifar100":  
    from Dictionary_Types.dic_cifar100 import DATA_NAME, NUM_CHANNELS, IMG_TYPE

    
import Classes.Baseblock as Baseblock
import Classes.ResNet18SS as ResNet18SS
import Classes.ResNet18CS as ResNet18CS
import Classes.DatasetManger as DatasetManger
import Classes.Server as Server
import Classes.Client as Client
##################################################################################
import torch, random, time, copy
from torch import nn
from pandas import DataFrame
import numpy as np
from datetime import date, datetime

###################Run training for model################# 
def run(Global,net_glob_client,net_glob_server, device, dataset_train,dataset_test,dict_users,dict_users_test,TcArray,TsArray):
    # this epoch is global epoch, also known as rounds
    start_time = time.time() 
    layersplit=[4,2]
    epochSplit=(int(EPOCHS/3),int(EPOCHS/3*2))
    for iter in range(EPOCHS):
        print("Global epoch:",iter)
        print(epochSplit,"/",iter>epochSplit[0],"/",iter<epochSplit[1],"/", iter>=epochSplit[1])
        if iter>=epochSplit[0] and iter<epochSplit[1]:
            print("Layer change")
            layersplit=[3,3]
        elif iter>=epochSplit[1]:
        #if True:
            print("Layer change")
            layersplit=[2,4]
        m = max(int(FRAC * NUM_USERS), 1)
        idxs_users = np.random.choice(range(NUM_USERS), m, replace = False)
        tempClientArray=[]
        # Sequential training/testing among clients      
        for idx in idxs_users:
            local = Client.Client(Global,LOCAL_EP,net_glob_client, idx, LR, device,layersplit, dataset_train = dataset_train, dataset_test = dataset_test, idxs = dict_users[idx], idxs_test = dict_users_test[idx])
            # Training ------------------
            w_client,tempArray = local.train(copy.deepcopy(net_glob_client).to(device),net_glob_server,device)
        
            # Testing -------------------
            local.evaluate(copy.deepcopy(net_glob_client).to(device),iter,net_glob_server,device)
            tempClientArray.append(tempArray)
            # copy weight to net_glob_client -- use to update the client-side model of the next client to be trained
            net_glob_client.load_state_dict(w_client)
        TcArray.append(tempClientArray)
        TsArray.append((time.time() - start_time)/60)
        
        

    return TcArray,TsArray

####################Sets up filename to save new output for results and time################
def setup_file_name():
    today = f"{date.today()}".replace("-","_")
    timeS=f"{datetime.now().strftime('%H:%M:%S')}".replace(":","_")
    program="SL"+"_D"+today+"_T"+timeS+DATA_NAME
    program = f"Results\\{program}_U{NUM_USERS}_E{EPOCHS}_e{LOCAL_EP}.xlsx"
    print(f"---------{program}----------")   
    return program
###################Save results of accuracy and time to xlsx file#################     
def save_results(Global,TsArray,TcArray,program):
    # Save output data to .excel file (we use for comparision plots)
    round_process = [i for i in range(1, len(Global.acc_train_collect)+1)]
    df = DataFrame({'round': round_process,'acc_train':Global.acc_train_collect, 'acc_test':Global.acc_test_collect, 'Gobal E Time (m)':TsArray, 'Local e Time per Client (m)': TcArray})     
    df.to_excel(program, sheet_name= "v1_test", index = False)   

###################Resnet client model and GPU parallel setup "if avaiable"#################
def setup_c_resnet(device,Global):
    net_glob_client = ResNet18CS.ResNet18_client_side(Global,NUM_CHANNELS,Baseblock.Baseblock,RESNETTYPE)
    if torch.cuda.device_count() > 1:
        print("We use",torch.cuda.device_count(), "GPUs")
        net_glob_client = nn.DataParallel(net_glob_client)   # to use the multiple GPUs; later we can change this to CPUs only     
    net_glob_client.to(device)
    return net_glob_client

###################Resnet server model and GPU parallel setup "if avaiable"#################
def setup_s_resnet(device,Global):
    net_glob_server = ResNet18SS.ResNet18_server_side(Global,Baseblock.Baseblock, RESNETTYPE, len(IMG_TYPE),NUM_CHANNELS) #7 is my numbr of classes
    if torch.cuda.device_count() > 1:
        print("We use",torch.cuda.device_count(), "GPUs")
        net_glob_server = nn.DataParallel(net_glob_server)   # to use the multiple GPUs 
    
    net_glob_server.to(device)  
    return net_glob_server

############################################################################################
###############################Start of program#############################################      
def main():
    global net_glob_client
    print("#############Setting up#############")
    #Name for the outputfile
    program=setup_file_name()
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"---------{device}----------")
    Global=Server.Server()
    
    print("#############Assign net_glob_client#############")
    net_glob_client = setup_c_resnet(device,Global)
        
    
    print("#############Assign net_glob_server and Server Functions#############")
    net_glob_server = setup_s_resnet(device,Global)
         
    
    
    print("#############Set up Dataset#############")
    dataset_train, dataset_test=DatasetManger.SetUpData(NUM_CHANNELS,DATA_NAME, IMG_TYPE)

    dict_users = DatasetManger.dataset_iid(dataset_train, NUM_USERS)
    dict_users_test = DatasetManger.dataset_iid(dataset_test, NUM_USERS)
    
    print("#############Start AI training#############")
    TcArray,TsArray=run(Global,net_glob_client,net_glob_server, device, dataset_train,dataset_test,dict_users,dict_users_test,TcArray,TsArray)
         
    
    print("Training and Evaluation completed!")    
    # Save output data to .excel file (we use for comparision plots)
    save_results(Global,TsArray,TcArray,program) 
    


main()












