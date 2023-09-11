import subprocess

splittype={
    1:"Central",
    2: "Split",
    3: "Dynamic"
    }

def runfile():
    print("Running...")
    process = subprocess.Popen(["python", "main.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # Wait for the process to complete and print any errors or exceptions
    process.communicate()
    print("Done\n")

def Start(arch,data,SPLITTYPE=3):
    print("Set to ",arch,data,splittype[SPLITTYPE])
    file=open(r"E:\Vehicular_Edge_Computing\Shared_Code\DynamicSplit\settings.py","r")
    collect=[]
    for line in file:
        if 'TRAINING_SORCE="' in line:
            collect.append(f'TRAINING_SORCE="{data}"\n')
        elif 'MODELTYPE="' in line:
            collect.append(f'MODELTYPE="{arch}"\n')
        elif 'SPLITTYPE='in line and "if" not in line:
            collect.append(f"SPLITTYPE={SPLITTYPE}\n")
        else:
            collect.append(line)
    file.close()
    file=open(r"E:\Vehicular_Edge_Computing\Shared_Code\DynamicSplit\settings.py","w")
    for i in range(len(collect)):
        file.writelines(collect[i])
    file.close()
    runfile()
    
        

#mnist10, fmnist10, cifar10, cifar100, ham10000, intelnet, IP102_FC_EC, IP102_FC, IP102_EC
#ResNet18, ResNet34, ResNet50, GoogleNet, MobileNet
#              Arch       Dataset

Start("ResNet18","IP102_FC_EC",SPLITTYPE=1)

Start("ResNet18","IP102_FC",SPLITTYPE=1)

Start("ResNet18","IP102_EC",SPLITTYPE=1)


Start("GoogleNet","IP102_FC_EC",SPLITTYPE=1)

Start("GoogleNet","IP102_FC",SPLITTYPE=1)

Start("GoogleNet","IP102_EC",SPLITTYPE=1)


Start("MobileNet","IP102_FC_EC",SPLITTYPE=1)

Start("MobileNet","IP102_FC",SPLITTYPE=1)

Start("MobileNet","IP102_EC",SPLITTYPE=1)

