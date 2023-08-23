from datasets import load_dataset
import os
import sys


def setup(n):
    name=n
    dataset = load_dataset(f"{name}")
    
    path1=r"data\Part1"+f"({name})"
    path2=r"data\Part2"f"({name})"
    try:
        os.mkdir(path1)
        os.mkdir(path2)
    except (FileExistsError):
        pass
    return dataset, path1, path2

def cifar100():
    name="cifar100"
    dataset,path1,path2=setup(name)
    #img.resize((64, 64))
    file1 = open(f"data/MyFile({name}).csv", "w")
    file1.write("image_id, fine_label, coarse_label\n")
    for i in range(len(dataset["train"])):
        img=dataset["train"][i]["img"]
        file_name=f"{path1}\\img_tr_{i}"
        print(f"Train: {i+1}/{len(dataset['train'])}: {file_name}", end="\r")
        img.save(f"{file_name}.jpg")
        coarse1=dataset["train"][i]["coarse_label"]
        fineLables1=dataset["train"][i]["fine_label"]
        file1.write(f"img_tr_{i},{fineLables1},{coarse1}\n")
    print()
    for i in range(len(dataset["test"])):
        img=dataset["test"][i]["img"]
        file_name=f"{path2}\\img_tr_{i}"
        print(f"Test: {i+1}/{len(dataset['train'])}: {file_name}", end="\r")
        img.save(f"{file_name}.jpg")
        coarse1=dataset["test"][i]["coarse_label"]
        fineLables1=dataset["test"][i]["fine_label"]
        file1.write(f"img_tr_{i},{fineLables1},{coarse1}\n")

def cifar10():
    name="cifar10"
    dataset,path1,path2=setup(name)
    #img.resize((64, 64))
    file1 = open(f"data/MyFile({name}).csv", "w")
    file1.write("image_id, fine_label, coarse_label\n")
    for i in range(len(dataset["train"])):
        img=dataset["train"][i]["img"]
        file_name=f"{path1}\\img_tr_{i}"
        print(f"Train: {i+1}/{len(dataset['train'])}: {file_name}", end="\r")
        img.save(f"{file_name}.jpg")
        fineLables=dataset["train"][i]["label"]
        file1.write(f"img_tr_{i},{fineLables}\n")
    print()
    for i in range(len(dataset["test"])):
        img=dataset["test"][i]["img"]
        file_name=f"{path2}\\img_te_{i}"
        print(f"Train: {i+1}/{len(dataset['test'])}: {file_name}", end="\r")
        img.save(f"{file_name}.jpg")
        fineLables=dataset["train"][i]["label"]
        file1.write(f"img_te_{i},{fineLables}\n")
    file1.close()
    ##################################################
    
def mnist():
    name="mnist"
    dataset,path1,path2=setup(name)
    #img.resize((64, 64))
    file1 = open(f"data/MyFile({name}).csv", "w")
    file1.write("image_id, fine_label\n")
    for i in range(len(dataset["train"])):
        img=dataset["train"][i]["image"]
        file_name=f"{path1}\\img_tr_{i}"
        print(f"Train: {i+1}/{len(dataset['train'])}: {file_name}", end="\r")
        img.save(f"{file_name}.jpg")
        fineLables=dataset["train"][i]["label"]
        file1.write(f"img_tr_{i},{fineLables}\n")
    print()
    for i in range(len(dataset["test"])):
        img=dataset["test"][i]["image"]
        file_name=f"{path2}\\img_tr_{i}"
        print(f"Test: {i+1}/{len(dataset['test'])}: {file_name}", end="\r")
        img.save(f"{file_name}.jpg")
        fineLables=dataset["test"][i]["label"]
        file1.write(f"img_tr_{i},{fineLables}\n")
    ##################################################
    
def fashion_mnist():
    name="fashion_mnist"
    dataset,path1,path2=setup(name)
    #img.resize((64, 64))
    file1 = open(f"data/MyFile({name}).csv", "w")
    file1.write("image_id, fine_label\n")
    for i in range(len(dataset["train"])):
        img=dataset["train"][i]["image"]
        file_name=f"{path1}\\img_tr_{i}"
        print(f"Train: {i+1}/{len(dataset['train'])}: {file_name}", end="\r")
        img.save(f"{file_name}.jpg")
        fineLables=dataset["train"][i]["label"]
        file1.write(f"img_tr_{i},{fineLables}\n")
    print()
    for i in range(len(dataset["test"])):
        img=dataset["test"][i]["image"]
        file_name=f"{path2}\\img_tr_{i}"
        print(f"Test: {i+1}/{len(dataset['test'])}: {file_name}", end="\r")
        img.save(f"{file_name}.jpg")
        fineLables=dataset["test"][i]["label"]
        file1.write(f"img_tr_{i},{fineLables}\n")
    ##################################################

def download(dataset):
    if "fmnist"==dataset:
        print("Downloading fashion_mnist")
        fashion_mnist()
    elif "mnist"==dataset:
        print("Downloading mnist")
        mnist()
    elif "cifar10"==dataset:
        print("Downloading cifar10")
        cifar10()
    elif "cifar100"==dataset:
        print("Downloading cifar100")
        cifar100()
    else:
        print("Dataset not found. Checkout README.txt")
        return
    print(dataset,"complete\n")

def run(arguments):
    if len(arguments)<=1:
        print("Missing arguments. Checkout README.txt")
        return
    for i in range(len(arguments)):
        if i!=0:
            download(arguments[i])
    print("All downloads Complete!")

def setup_run(arguments):
    if len(arguments)<1:
        print("Missing arguments. Checkout README.txt")
        return
    for i in range(len(arguments)):
        download(arguments[i])
    print("All downloads Complete!")
if __name__ == '__main__':
    run(sys.argv)


