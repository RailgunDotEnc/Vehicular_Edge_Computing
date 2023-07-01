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
            

    def forward2_4(self,x):
        resudial1 = F.relu(self.layers[0](x))
        out1 = self.layers[1](resudial1)
        out1 = out1 + resudial1 # adding the resudial inputs -- downsampling not required in this layer
        resudial2 = F.relu(out1)
        
        return resudial2
    
    def forward3_3(self,x):
        resudial1 = F.relu(self.layers[0](x))
        out1 = self.layers[1](resudial1)
        out1 = out1 + resudial1 # adding the resudial inputs -- downsampling not required in this layer
        resudial2 = F.relu(out1)
        
        out2 = self.layers[2](resudial2)
        out2 = out2 + resudial2          # adding the resudial inputs -- downsampling not required in this layer
        x3 = F.relu(out2)
        
        return x3
    
    def forward4_2(self,x):
        resudial1 = F.relu(self.layers[0](x))
        out1 = self.layers[1](resudial1)
        out1 = out1 + resudial1 # adding the resudial inputs -- downsampling not required in this layer
        resudial2 = F.relu(out1)
        
        out2 = self.layers[2](resudial2)
        out2 = out2 + resudial2          # adding the resudial inputs -- downsampling not required in this layer
        x3 = F.relu(out2)
        x4 = self.layers[3](x3)
        
        return x4
    def forward5_1(self,x):
        resudial1 = F.relu(self.layers[0](x))
        out1 = self.layers[1](resudial1)
        out1 = out1 + resudial1 # adding the resudial inputs -- downsampling not required in this layer
        resudial2 = F.relu(out1)
        
        out2 = self.layers[2](resudial2)
        out2 = out2 + resudial2          # adding the resudial inputs -- downsampling not required in this layer
        x3 = F.relu(out2)
        x4 = self.layers[3](x3)
        x5 = self.layers[4](x4)
        return x5
        
        
    
    def forward(self, x, Layer_Count):
        #Variable for layers to send to server
        volly=None
        if Layer_Count!=self.Layer_Count:
            if Layer_Count[0]>self.Layer_Count[0] and Layer_Count[1]>self.Layer_Count[1]:
                self.addLayer(x,Layer_Count)
            else:
                volly=self.removeLayer(x,Layer_Count)
            
        if self.Layer_Count==[2,4]:
            layer=self.forward2_4(x)
        elif self.Layer_Count==[3,3]:
            layer=self.forward3_3(x)
        elif self.Layer_Count==[4,2]:
            layer=self.forward4_2(x)
        return layer,volly
    
    def addLayer(self,x,Layer_Count):
        print("Layer(s) added to Client")
        print("f{self.Layer_Count[0]} => {Layer_Count[0]}")
        self.Layer_Count=Layer_Count
        pass
    def removeLayer(self,x,Layer_Count):
        print("Layer(s) removed from Client")
        print(f"{self.Layer_Count[0]} => {Layer_Count[0]}")
        diff=self.Layer_Count[0]-Layer_Count[0]
        volly=[None,None,None,None,None,None]
        for i in range(diff):
            volly[self.Layer_Count[0]-diff+i]=self.layers[self.Layer_Count[0]-diff+i]
            self.layers[(len(self.layers)-diff)+i]=None
        self.Layer_Count=Layer_Count
        return volly
        
        
        
        
        
        
        
        