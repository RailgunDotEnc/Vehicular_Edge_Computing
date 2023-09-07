import subprocess


def runfile():
    print("Running...")
    process = subprocess.Popen(["python", "main.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # Wait for the process to complete and print any errors or exceptions
    process.communicate()
    print("Done\n")

def make_changes(arch,data,DynamicSplit=False):
    print("Set to ",arch,data,DynamicSplit)
    file=open(r"E:\Vehicular_Edge_Computing\Shared_Code\DynamicSplit\settings.py","r")
    collect=[]
    for line in file:
        if 'TRAINING_SORCE="' in line:
            collect.append(f'TRAINING_SORCE="{data}"\n')
        elif 'MODELTYPE="' in line:
            collect.append(f'MODELTYPE="{arch}"\n')
        elif 'ACTIVATEDYNAMIC='in line:
            collect.append(f"ACTIVATEDYNAMIC={DynamicSplit}\n")
        else:
            collect.append(line)
    file.close()
    file=open(r"E:\Vehicular_Edge_Computing\Shared_Code\DynamicSplit\settings.py","w")
    for i in range(len(collect)):
        file.writelines(collect[i])
    file.close()
        

#mnist10, fmnist10, cifar10, cifar100, ham10000, intelnet, IP102_FC_EC, IP102_FC, IP102_EC
#ResNet18, ResNet34, ResNet50, GoogleNet, MobileNet
#              Arch       Dataset

#make_changes("ResNet18","IP102_FC_EC")
#runfile()
#make_changes("ResNet18","IP102_FC")
#runfile()
#make_changes("ResNet18","IP102_EC")
#runfile()

#make_changes("GoogleNet","IP102_FC_EC")
#runfile()
#make_changes("GoogleNet","IP102_FC")
#runfile()
#make_changes("GoogleNet","IP102_EC")
#runfile()

make_changes("MobileNet","IP102_FC_EC")
runfile()
make_changes("MobileNet","IP102_FC")
runfile()
make_changes("MobileNet","IP102_EC")
runfile()
