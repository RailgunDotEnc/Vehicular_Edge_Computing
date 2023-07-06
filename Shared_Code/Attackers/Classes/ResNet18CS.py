import torch.nn.functional as F
from torch import nn
import math

# Model at client side
class ResNet18_client_side(nn.Module):
    def __init__(self,global_server,channels,block,num_layers):
        super(ResNet18_client_side, self).__init__()
        self.Layer_Count=[4,2]
        self.Global=global_server
        self.layer1 = nn.Sequential (
                nn.Conv2d(channels, 64, kernel_size = 7, stride = 2, padding = 3, bias = False),
                nn.BatchNorm2d(64),
                nn.ReLU (inplace = True),
                nn.MaxPool2d(kernel_size = 3, stride = 2, padding =1),
            )
        self.layer2 = nn.Sequential  (
                nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1, bias = False),
                nn.BatchNorm2d(64),
                nn.ReLU (inplace = True),
                nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1),
                nn.BatchNorm2d(64),              
            )
        self.layer3 = nn.Sequential (
                nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1),
                nn.BatchNorm2d(64),
                nn.ReLU (inplace = True),
                nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1),
                nn.BatchNorm2d(64),       
                )
        self.layer4 = self._layer(block, 128, num_layers[0],64, stride = 2)
        self.layer5 =self._layer(block, 256, num_layers[1],128, stride = 2)
        
        self.layers=[self.layer1,self.layer2,self.layer3,self.layer4,None,None]
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
    
    def _layer(self, block, planes, num_layers,input_planes, stride = 2):
        dim_change = None
        if stride != 1 or planes != input_planes * block.expansion:
            dim_change = nn.Sequential(nn.Conv2d(input_planes, planes*block.expansion, kernel_size = 1, stride = stride),
                                       nn.BatchNorm2d(planes*block.expansion))
        netLayers = []
        netLayers.append(block(input_planes, planes, stride = stride, dim_change = dim_change))
        input_planes = planes * block.expansion
        for i in range(1, num_layers):
            netLayers.append(block(input_planes, planes))
            input_planes = planes * block.expansion
        return nn.Sequential(*netLayers)
            
    
    def forward(self, x, Layer_Count,volly=None):
        if self.layers[0]!=None:
            #Variable for layers to send to server
            resudial1 = F.relu(self.layers[0](x))
            x = self.layers[1](resudial1)
            x = x + resudial1 # adding the resudial inputs -- downsampling not required in this layer
        if self.layers[1]!=None:
            resudial2 = F.relu(x)
            x = self.layers[2](resudial2)
            x = x + resudial2          # adding the resudial inputs -- downsampling not required in this layer
        if self.layers[2]!=None:
            x = F.relu(x)
        if self.layers[3]:
            x = self.layers[3](x)
        if self.layers[4]:
            x = self.layers[4](x)
        return x,volly
        
    
    def addLayer(self,x,Layer_Count):
        print("Layer(s) added to Client")
        print("f{self.Layer_Count[0]} => {Layer_Count[0]}")
        self.Layer_Count=Layer_Count
        pass
    def removeLayer(self,x,Layer_Count):
        print("Layer(s) removed from Client")
        print(f"{self.Layer_Count[0]} => {Layer_Count[0]}")
        self.Layer_Count=Layer_Count
        diff=self.Layer_Count[0]-Layer_Count[0]
        volly=[None,None,None,None,None,None]
        for i in range(diff):
            volly[self.Layer_Count[0]-diff+i]=self.layers[self.Layer_Count[0]-diff+i]
            self.layers[(len(self.layers)-diff)+i]=None
        self.Layer_Count=Layer_Count
        return volly
    
    def is_attacker(self):
        return False
    
        
        
        
        
        
        
        
        
        
        