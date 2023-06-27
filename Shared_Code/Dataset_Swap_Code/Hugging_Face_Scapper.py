from datasets import load_dataset
import os

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
        print(f"Train: {i+1}/{len(dataset['train'])}: {file_name}")
        img.save(f"{file_name}.jpg")
        coarse1=dataset["train"][i]["coarse_label"]
        fineLables1=dataset["train"][i]["fine_label"]
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
        print(f"Train: {i+1}/{len(dataset['train'])}: {file_name}")
        img.save(f"{file_name}.jpg")
        fineLables=dataset["train"][i]["label"]
        file1.write(f"img_tr_{i},{fineLables}\n")
        
    for i in range(len(dataset["test"])):
        img=dataset["test"][i]["img"]
        file_name=f"{path2}\\img_te_{i}"
        print(f"Train: {i+1}/{len(dataset['test'])}: {file_name}")
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
        print(f"Train: {i+1}/{len(dataset['train'])}: {file_name}")
        img.save(f"{file_name}.jpg")
        fineLables=dataset["train"][i]["label"]
        file1.write(f"img_tr_{i},{fineLables}\n")
        
    for i in range(len(dataset["test"])):
        img=dataset["test"][i]["image"]
        file_name=f"{path1}\\img_tr_{i}"
        print(f"Train: {i+1}/{len(dataset['test'])}: {file_name}")
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
        print(f"Train: {i+1}/{len(dataset['train'])}: {file_name}")
        img.save(f"{file_name}.jpg")
        fineLables=dataset["train"][i]["label"]
        file1.write(f"img_tr_{i},{fineLables}\n")
        
    for i in range(len(dataset["test"])):
        img=dataset["test"][i]["image"]
        file_name=f"{path1}\\img_tr_{i}"
        print(f"Train: {i+1}/{len(dataset['test'])}: {file_name}")
        img.save(f"{file_name}.jpg")
        fineLables=dataset["test"][i]["label"]
        file1.write(f"img_tr_{i},{fineLables}\n")
    ##################################################

print("Running fashion_mnist")
fashion_mnist()
print("Running mnist")
mnist()
print("Running cifar10")
#cifar10()
print("Running cifar100")
#cifar100()



