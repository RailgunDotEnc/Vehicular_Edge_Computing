RESNETTYPE=[2,2,2,2] #34 up [3,4,6,3], #18 [2,2,2,2]
NUM_USERS = 1
EPOCHS = 100
LOCAL_EP=1
FRAC = 1
LR = 0.0001
#mnist10, fmnist10, cifar10, cifar100, ham10000, intelnet, IP102_FC_EC, IP102_FC, IP102_EC
TRAINING_SORCE="IP102_EC"
EPOCHSPLIT=3
#CL: 1 (Make client 1 and local_EP 1), SL: 2, DSL: 3 
SPLITTYPE=3
if SPLITTYPE==1:
    NUM_USERS=1
    LOCAL_EP=1
    
#ResNet18, ResNet34, ResNet50, GoogleNet, MobileNet
MODELTYPE="MobileNet"
clientlayers=2
NOISE=False

if TRAINING_SORCE=="mnist10":
    from Dictionary_Types.dic_mnist10 import DATA_NAME, NUM_CHANNELS, IMG_TYPE
elif TRAINING_SORCE=="fmnist10":
    from Dictionary_Types.dic_fmnist10 import DATA_NAME, NUM_CHANNELS, IMG_TYPE
elif TRAINING_SORCE=="cifar10":
    from Dictionary_Types.dic_cifar10 import DATA_NAME, NUM_CHANNELS, IMG_TYPE
elif TRAINING_SORCE=="cifar100":  
    from Dictionary_Types.dic_cifar100 import DATA_NAME, NUM_CHANNELS, IMG_TYPE
elif TRAINING_SORCE=="ham10000":  
    from Dictionary_Types.dic_ham10000 import DATA_NAME, NUM_CHANNELS, IMG_TYPE
elif TRAINING_SORCE=="intelnet":  
    from Dictionary_Types.dic_intelnet import DATA_NAME, NUM_CHANNELS, IMG_TYPE
elif TRAINING_SORCE=="IP102_FC_EC":  
    from Dictionary_Types.dic_IP102_FC_EC import DATA_NAME, NUM_CHANNELS, IMG_TYPE
elif TRAINING_SORCE=="IP102_FC":  
    from Dictionary_Types.dic_IP102_FC import DATA_NAME, NUM_CHANNELS, IMG_TYPE
elif TRAINING_SORCE=="IP102_EC":  
    from Dictionary_Types.dic_IP102_EC import DATA_NAME, NUM_CHANNELS, IMG_TYPE
