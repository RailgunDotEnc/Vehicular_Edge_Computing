import subprocess

splittype={
    1:"Central",
    2: "Split",
    3: "Dynamic"
    }

def runfile(file):
    print("Running...")
    if file=="main":
        process = subprocess.Popen(["python", "main.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    else:
        process = subprocess.Popen(["python", "FL_main.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # Wait for the process to complete and print any errors or exceptions
    process.communicate()
    print("Done\n")

def Start(file,arch,data,SPLITTYPE=3):
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
    runfile(file)
    
        

#mnist10, fmnist10, cifar10, cifar100, ham10000, intelnet, IP102_FC_EC, IP102_FC, IP102_EC
#ResNet18, ResNet34, ResNet50, GoogleNet, MobileNet
#              Arch       Dataset

Start("FL_main","ResNet18","IP102_FC_EC")

Start("FL_main","ResNet18","IP102_FC")

Start("FL_main","ResNet18","IP102_EC")


Start("FL_main","GoogleNet","IP102_FC_EC")

Start("FL_main","GoogleNet","IP102_FC")

Start("FL_main","GoogleNet","IP102_EC")


Start("FL_main","MobileNet","IP102_FC_EC")

Start("FL_main","MobileNet","IP102_FC")

Start("FL_main","MobileNet","IP102_EC")

