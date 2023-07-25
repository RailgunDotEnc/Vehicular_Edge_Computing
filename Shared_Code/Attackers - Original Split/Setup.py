import os


def makedir(path):
    try:
        os.mkdir(path)
    except Exception as e:
        print(e)

#Folder for Huggingface datascrapper
makedir("Data")

makedir("Results")
