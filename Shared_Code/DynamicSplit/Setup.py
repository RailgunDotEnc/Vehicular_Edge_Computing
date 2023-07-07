import os
import subprocess
import sys
import Hugging_Face_Scapper

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install -r", package])
    
def runHuggingface(datasets):
    args=""
    for i in range(len(datasets)):
        if i!=len(datasets)-1:
            args=args+datasets[i]+" "
        else:
            args=args+datasets[i]
    print("Downloading:", args)
    args=["Import",args]
    Hugging_Face_Scapper.run(args)


def makedir(path):
    try:
        os.mkdir(path)
    except Exception as e:
        print(e)

#Folder for Huggingface datascrapper
makedir("Data")

makedir("Results")

#Download requirements
#install("requirements.txt")

#Which dataset to download
datasets=[]
while True:
    user=int(input("Enter number (1 by 1) \n(1) mnist, (2) fmnist, (3) cifar10, (4) cifar100 (5) Submit: "))
    if user==5:
        break
    elif user==1:
        datasets.append("mnist")
    elif user==2:
        datasets.append("fmnist")
    elif user==3:
        datasets.append("cifar10")
    elif user==4:
        datasets.append("cifar100")
        
if datasets!=[]:
    runHuggingface(datasets)

print("\n#################")
print("Setup Complete :)")
print("#################")
    
    
