#=============================================================================
# Split learning: ResNet18
# ============================================================================
#Code imports
from settings import RESNETTYPE, NUM_USERS, EPOCHS, LOCAL_EP, FRAC, LR, TRAINING_SORCE, ACTIVATEDYNAMIC, MODELTYPE,clientlayers,NOISE,DATA_NAME, NUM_CHANNELS, IMG_TYPE
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import label_binarize

import Classes.DatasetManger as DatasetManger    
import Classes.Server_modi_V1 as Server
import Classes.Client as Client
#Resnet
import Classes.Baseblock as Baseblock
import Classes.ResNet18SS as ResNet18SS
import Classes.ResNet18CS as ResNet18CS


#Google
import Classes.GoogleBlocks as GoogleBlocks
import Classes.GoogleNetCS as GoogleNetCS
import Classes.GoogleNetSS as GoogleNetSS

#MobileNet
import Classes.MobileNetBlocks as MobileNetBlocks
import Classes.MobileNetCS as MobileNetCS
import Classes.MobileNetSS as MobileNetSS
##################################################################################
import torch, random, time, copy
from torch import nn
from pandas import DataFrame
import numpy as np
from datetime import date, datetime


def overfitting(epoch,Global):
    
    if epoch > 0:
        diff=Global.loss_test_collect[epoch-1]-Global.loss_test_collect[epoch]
        if diff <0:
            print("Loss warning")
            Global.tick=Global.tick+1
        else:
            Global.tick=0
    if Global.tick>2:
        return True
    else:
        return False

def changelayer(layersplit):
    total=layersplit[0]+layersplit[1]
    rand=random.randint(2, total-2)
    new_layersplit=[rand,total-rand]
    while new_layersplit==layersplit:
        rand=random.randint(2, total-2)
        new_layersplit=[rand,total-rand] 
    print("Layer update: ",[rand,total-rand])
    return [rand,total-rand]

###################Run training for model################# 
def run(Global,net_glob_client,net_glob_server, device, dataset_train,dataset_test,dict_users,dict_users_test,Arch_Layers):
    # this epoch is global epoch, also known as rounds
    global w_client
    TsArray=[]
    TcArray=[]
    SplArray=[]
    ClientArray=[]
    start_time = time.time()
    
    layersplit=Arch_Layers.copy()
    
    #Start creating model
    for iter in range(EPOCHS):
        print("Global epoch:",iter)
        
        m = max(int(FRAC * NUM_USERS), 1)
        idxs_users = np.random.choice(range(NUM_USERS), m, replace = False)
        tempClientArray=[]
        tempClientSplitArray=[]
        tempCArray=[]
        
        # Sequential training/testing among clients      
        for idx in idxs_users:
            #Test change in layer
            print("\nBase Layer:",layersplit)
            rand=random.randint(0,1)
            if rand == 1 and ACTIVATEDYNAMIC==True:
                layersplit=changelayer(layersplit)
            #Save Layers per client
            tempClientSplitArray.append(layersplit)
            tempCArray.append(idx)
            
            #Set up client and check for layers needed
            local = Client.Client(Global,LOCAL_EP,layersplit,net_glob_client, idx, LR, device, dataset_train = dataset_train, dataset_test = dataset_test, idxs = dict_users[idx], idxs_test = dict_users_test[idx])
            C_layers,Layer_Count=local.check4update(net_glob_client,net_glob_server)
            net_glob_client.layers=C_layers
            net_glob_client.Layer_Count=Layer_Count
            
            # Training ------------------
            w_client,tempArray = local.train(copy.deepcopy(net_glob_client).to(device),net_glob_server,device,NOISE)
            
            # Testing -------------------
            local.evaluate(copy.deepcopy(net_glob_client).to(device),iter,net_glob_server,device,NOISE)
                        
            tempClientArray.append(tempArray)
            # copy weight to net_glob_client -- use to update the client-side model of the next client to be trained
            net_glob_client.load_state_dict(w_client)
            
        #Save times
        ClientArray.append(tempCArray)
        SplArray.append(tempClientSplitArray)
        TcArray.append(tempClientArray)
        TsArray.append((time.time() - start_time)/60)
        
        #Stop code if model starts over fitting
        #if overfitting(iter, Global):
        #    break        
            
    return TcArray,TsArray,SplArray,ClientArray

####################Sets up filename to save new output for results and time################
def setup_file_name():
    today = f"{date.today()}".replace("-","_")
    timeS=f"{datetime.now().strftime('%H:%M:%S')}".replace(":","_")
    if NUM_USERS==1:
        program="CL"+MODELTYPE+"_D"+today+"_T"+timeS+DATA_NAME+f"_U{NUM_USERS}_E{EPOCHS}_e{LOCAL_EP}_Noise{NOISE}.xlsx"
    elif ACTIVATEDYNAMIC==True:
        program="DSL_"+MODELTYPE+"_D"+today+"_T"+timeS+DATA_NAME+f"_U{NUM_USERS}_E{EPOCHS}_e{LOCAL_EP}_Noise{NOISE}.xlsx"
    elif ACTIVATEDYNAMIC==False:
        program="SL_"+MODELTYPE+"_D"+today+"_T"+timeS+DATA_NAME+f"_U{NUM_USERS}_E{EPOCHS}_e{LOCAL_EP}_Noise{NOISE}.xlsx"
    print(f"---------{program}----------")   
    program = f"Results\\{program}"
    return program

################### ROC Path #########################
def setup_roc_path():
    today = f"{date.today()}".replace("-","_")
    timeS=f"{datetime.now().strftime('%H:%M:%S')}".replace(":","_")
    if NUM_USERS==1:
        roc_path = "CL"+MODELTYPE+"_D"+today+"_T"+timeS+DATA_NAME+f"_U{NUM_USERS}_E{EPOCHS}_e{LOCAL_EP}_Noise{NOISE}.png"
    elif ACTIVATEDYNAMIC==True:
        roc_path = "DSL_"+MODELTYPE+"_D"+today+"_T"+timeS+DATA_NAME+f"_U{NUM_USERS}_E{EPOCHS}_e{LOCAL_EP}_Noise{NOISE}.png"
    elif ACTIVATEDYNAMIC==False:
        roc_path = "SL_"+MODELTYPE+"_D"+today+"_T"+timeS+DATA_NAME+f"_U{NUM_USERS}_E{EPOCHS}_e{LOCAL_EP}_Noise{NOISE}.png"
    print(f"---------{roc_path}----------")   
    roc_path = f"Results\\{roc_path}"
    return roc_path
####################################################

###################Save results of accuracy and time to xlsx file#################     
def save_results(Global,TsArray,TcArray,program,SplArray,ClientArray):
    print("#########Saving Results############")
    # Save output data to .excel file (we use for comparision plots)
    round_process = [i for i in range(1, len(Global.acc_train_collect)+1)]
    
    df = DataFrame({'round': round_process,'acc_train':Global.acc_train_collect, 'acc_test':Global.acc_test_collect, "Loss":Global.loss_test_collect, "Precision":Global.precision_test_collect, "Recall":Global.recall_test_collect, "F1":Global.f1_test_collect, "G-mean (micro)":Global.gmean_micro_test_collect, "G-mean (macro)":Global.gmean_macro_test_collect, 'Gobal E Time (m)':TsArray, 'Local e Time per Client (m)': TcArray,'Split Count':SplArray, "Client":ClientArray})     
      
    df.to_excel(program, sheet_name= "v1_test", index = False)   

###################Resnet client model and GPU parallel setup "if avaiable"#################
def setup_c_resnet(device,Global):
    Arch_Layers=[2,4]
    if MODELTYPE=="ResNet50":
        Arch_Layers=[clientlayers,6-clientlayers]
        print("Using Bottleneck")
        net_glob_client = ResNet18CS.ResNet18_client_side(Global,NUM_CHANNELS,Baseblock.Bottleneck,RESNETTYPE,Arch_Layers)
    elif MODELTYPE=="ResNet18" or MODELTYPE=="ResNet34":
        print("Using BaseBlock")
        Arch_Layers=[clientlayers,6-clientlayers]
        net_glob_client = ResNet18CS.ResNet18_client_side(Global,NUM_CHANNELS,Baseblock.Baseblock,RESNETTYPE,Arch_Layers)
    elif MODELTYPE=="GoogleNet":
        Arch_Layers=[clientlayers,5-clientlayers]
        net_glob_client = GoogleNetCS.GoogLeNetClient(NUM_CHANNELS,len(IMG_TYPE),GoogleBlocks.conv_block,GoogleBlocks.Inception_block,Arch_Layers)
    elif MODELTYPE=="MobileNet":
        Arch_Layers=[clientlayers,16-clientlayers]
        net_glob_client = MobileNetCS.MobileNetV3Client("large", NUM_CHANNELS,len(IMG_TYPE),MobileNetBlocks.MobileNetV3Block,MobileNetBlocks.BNeck,Arch_Layers)
    
    net_glob_client.to(device)
    return net_glob_client, Arch_Layers

###################Resnet server model and GPU parallel setup "if avaiable"#################
def setup_s_resnet(device,Global,Arch_Layers):
    if MODELTYPE=="ResNet50":
        print("Using Bottleneck")
        net_glob_server = ResNet18SS.ResNet18_server_side(Global,Baseblock.Bottleneck, RESNETTYPE, len(IMG_TYPE),NUM_CHANNELS,Arch_Layers) #7 is my numbr of classes
    if MODELTYPE=="ResNet18" or MODELTYPE=="ResNet34":
        print("Using BaseBlock")
        net_glob_server = ResNet18SS.ResNet18_server_side(Global,Baseblock.Baseblock, RESNETTYPE, len(IMG_TYPE),NUM_CHANNELS,Arch_Layers) #7 is my numbr of classes
    if MODELTYPE=="GoogleNet":
        net_glob_server = GoogleNetSS.GoogLeNetServer(NUM_CHANNELS,len(IMG_TYPE),GoogleBlocks.Inception_block,Arch_Layers)
    if MODELTYPE=="MobileNet":
        net_glob_server = MobileNetSS.MobileNetV3Server("large", NUM_CHANNELS,len(IMG_TYPE),MobileNetBlocks.MobileNetV3Block,MobileNetBlocks.BNeck,Arch_Layers)
    
    net_glob_server.to(device)  
    return net_glob_server

#################################### ROC Plot###################
def plot_roc_curves(Global, num_classes=5):
    # Convert true_labels and pred_probs to numpy arrays
    roc_path = setup_roc_path()
    true_labels = Global.get_true_labels()
    pred_probs = Global.get_pred_probs()

    true_labels = np.array(true_labels)
    pred_probs = np.array(pred_probs)

    # Binarize the labels for one-vs-all ROC curve
    true_labels = label_binarize(true_labels, classes=[0, 1, 2, 3, 4])

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(true_labels[:, i], pred_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curves
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], label=f'ROC curve of class {i} (area = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.savefig(roc_path)  # Save the plot
#     plt.show()

############################################################################################
###############################Start of program#############################################      
def main():
    global net_glob_client
    print("#############Setting up#############")
    #Name for the outputfile
    program=setup_file_name()
     
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
    net_glob_client,Arch_Layers = setup_c_resnet(device,Global)
        
    
    print("#############Assign net_glob_server and Server Functions#############")
    net_glob_server = setup_s_resnet(device,Global,Arch_Layers)
    
    print("#############Set up Dataset#############")
    dataset_train, dataset_test=DatasetManger.SetUpData(NUM_CHANNELS,DATA_NAME, IMG_TYPE)

    dict_users = DatasetManger.dataset_iid(dataset_train, NUM_USERS)
    dict_users_test = DatasetManger.dataset_iid(dataset_test, NUM_USERS)
    
    print("#############Start AI training#############")
    TcArray,TsArray,SplArray,ClientArray=run(Global,net_glob_client,net_glob_server, device, dataset_train, dataset_test, dict_users, dict_users_test, Arch_Layers)
         
    
    print("Training and Evaluation completed!")    
    # Save output data to .excel file (we use for comparision plots)
    save_results(Global,TsArray,TcArray,program,SplArray,ClientArray) 
    
    plot_roc_curves(Global, num_classes=len(IMG_TYPE))
    
main()
