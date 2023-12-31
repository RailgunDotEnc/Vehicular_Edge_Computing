import torch.nn.functional as F
from torch import nn
import math

# Model at client side
class GoogLeNetClient(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000,conv_block=None,Inception_block=None, Layers=None):
        super(GoogLeNetClient, self).__init__()
        self.Layer_Count=Layers.copy()
        self.layers=[]
        for i in range(5):
            if i<self.Layer_Count[0]:
                self.layers.append(f"layer{i+1}")
            else:
                self.layers.append(None)
        print(self.layers)
        self.conv1 = conv_block(in_channels=in_channels, out_channels = 64, kernel_size=7, stride =2, padding = 3)
        
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = conv_block(64, 192, kernel_size =3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception3a = Inception_block(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception_block(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.inception4a = Inception_block(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception_block(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception_block(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception_block(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception_block(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        print(self.state_dict().keys())
        
        
    def forward(self, x, Layer_Count,volly=None):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        #Layer3
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
        return x,volly
    
    def get_weights(self,client_dict,layers):
        keys=list(client_dict.keys())
        volly={}
        for i in range(len(keys)):
            for j in range(len(layers)):
                if f"inception{layers[j]}" in keys[i]:
                    #print(f"Moving layer {keys[i]}")
                    volly[f"{ keys[i]}"]=client_dict[keys[i]]
        return(volly)
    
    def activate_layers(self,layers):
        print("Client activate:",layers)
        all_layers=["layer1","layer2","layer3","layer4"]
        for i in range(len(layers)):
            self.layers[layers[i]-1]=all_layers[layers[i]-1]
            
        new_layer_count=0
        for i in range(len(self.layers)):
            if self.layers[i]!=None:
                new_layer_count=new_layer_count+1
        print(self.layers)
        self.Layer_Count=[new_layer_count,5-new_layer_count]
            
    def deactivate_layers(self,layers):
        print("Client Deactivate:",layers)
        for i in range(len(layers)):
            self.layers[layers[i]-1]=None
        new_layer_count=0
        for i in range(len(self.layers)):
            if self.layers[i]!=None:
                new_layer_count=new_layer_count+1
        print(self.layers)
        self.Layer_Count=[new_layer_count,5-new_layer_count]
        
        
        
        
        
        
        
        
        