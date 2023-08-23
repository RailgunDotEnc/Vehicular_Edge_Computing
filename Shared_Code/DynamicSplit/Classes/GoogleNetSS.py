import torch.nn.functional as F
from torch import nn
import math

# Model at client side



class GoogLeNetServer(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000,Inception_block=None):
        super(GoogLeNetServer, self).__init__()
        self.Layer_Count=[2,4]
        self.layers=[]
        for i in range(6):
            if i>=self.Layer_Count[0]:
                self.layers.append(f"layer{i+1}")
            else:
                self.layers.append(None)
        print(self.layers)
        self.inception3a = Inception_block(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception_block(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.inception4a = Inception_block(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception_block(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception_block(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception_block(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception_block(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.inception5a = Inception_block(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception_block(832, 384, 192, 384, 48, 128, 128)
        
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=1)
        self.dropout = nn.Dropout(p=0.4)
        self.fc1 = nn.Linear(1024,num_classes)
        #self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x,Layer_Count,volly=None):
        if self.layers[2]!=None:
            x = self.inception3a(x)
            x = self.inception3b(x)
            x = self.maxpool3(x)
        if self.layers[3]:
            x = self.inception4a(x)
            x = self.inception4b(x)
            x = self.inception4c(x)
            x = self.inception4d(x)
            x = self.inception4e(x)
            x = self.maxpool4(x)
        if self.layers[4]:
            x = self.inception5a(x)
            x = self.inception5b(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0],-1)
        x = self.dropout(x)
        x = self.fc1(x)
        #x = self.softmax
        return x
        
    def get_weights(self,client_dict,layers):
            keys=list(client_dict.keys())
            volly={}
            for i in range(len(keys)):
                for j in range(len(layers)):
                    if f"layer{layers[j]}." in keys[i]:
                        volly[f"{keys[i]}"]=client_dict[keys[i]]
            self.Saved_Layers=volly.copy()
            return(volly)
        
    def activate_layers(self,layers):
        print("Server activate:",layers)
        all_layers=["layer2","layer3","layer4","layer5","layer6"]
        for i in range(len(layers)):
            self.layers[layers[i]-1]=all_layers[layers[i]-1]
            
        new_layer_count=0
        for i in range(len(self.layers)):
            if self.layers[i]!=None:
                new_layer_count=new_layer_count+1
        print(self.layers)
        self.Layer_Count=[6-new_layer_count,new_layer_count]
            
    def deactivate_layers(self,layers):
        print("Server Deactivate:",layers)
        for i in range(len(layers)):
            self.layers[layers[i]-1]=None
        new_layer_count=0
        for i in range(len(self.layers)):
            if self.layers[i]!=None:
                new_layer_count=new_layer_count+1
        print(self.layers)
        self.Layer_Count=[6-new_layer_count,new_layer_count]
        
        
        
        
        
        
        
        