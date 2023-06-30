import torch.nn.functional as F
from torch import nn
import math

# Model at server side
class ResNet18_server_side(nn.Module):
    def __init__(self,global_server, block, num_layers, classes,channels):
        super(ResNet18_server_side, self).__init__()
        self.Global=global_server
        self.Layer_Count=[4,2]
        self.layer2 = nn.Sequential  (
                nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1, bias = False),
                nn.BatchNorm2d(64),
                nn.ReLU (inplace = True),
                nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1),
                nn.BatchNorm2d(64),              
            )
        #####################################################
        #OG split
        self.layer3 = nn.Sequential (
                nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1),
                nn.BatchNorm2d(64),
                nn.ReLU (inplace = True),
                nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1),
                nn.BatchNorm2d(64),       
                ) 

        self.layer4 = self._layer(block, 128, num_layers[0],64, stride = 2)
        self.layer5 = self._layer(block, 256, num_layers[1],128, stride = 2)
        self.layer6 = self._layer(block, 512, num_layers[2],256, stride = 2)
        
        self.layers=[self.layer5,self.layer6]
        
        self. averagePool = nn.AvgPool2d(kernel_size = 7, stride = 1)
        self.fc = nn.Linear(512 * block.expansion, classes)
        
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
        out2 = self.layers[0](x)
        out2 = out2 + x          # adding the resudial inputs -- downsampling not required in this layer
        x3 = F.relu(out2)
        
        x4 = self. layers[1](x3)
        x5 = self.layers[2](x4)
        x6 = self.layers[3](x5)
        
        x7 = F.avg_pool2d(x6, 2)
        x8 = x7.view(x7.size(0), -1) 
        y_hat =self.fc(x8)
        
        return y_hat
    
    def forward3_3(self,x3):
        x4 = self.layers[0](x3)
        x5 = self.layers[1](x4)
        x6 = self.layers[2](x5)
        x7 = F.avg_pool2d(x6, 2)
        x8 = x7.view(x7.size(0), -1) 
        y_hat =self.fc(x8)
        return y_hat
    
    def forward4_2(self,x4):
        x5 = self.layers[0](x4)
        x6 = self.layers[1](x5)
        x7 = F.avg_pool2d(x6, 2)
        x8 = x7.view(x7.size(0), -1) 
        y_hat =self.fc(x8)
        return y_hat
    
    def forward(self, x,Layer_Count,volly=None):   
        if Layer_Count!=self.Layer_Count:
            if Layer_Count[0]>self.Layer_Count[0] and Layer_Count[1]>self.Layer_Count[1]:
                self.removeLayer(x,Layer_Count)
            else:
                self.addLayer(x,Layer_Count,volly)               
        if self.Layer_Count==[2,4]:
            y_hat=self.forward2_4(x)
        elif self.Layer_Count==[3,3]:
            y_hat=self.forward3_3(x)
        elif self.Layer_Count==[4,2]:
            y_hat=self.forward4_2(x)
        return y_hat
    
    def addLayer(self,x,Layer_Count,volly):
        print("Layer(s) added to Server")
        print(f"{self.Layer_Count[1]} => {Layer_Count[1]}")
        diff=Layer_Count[1]-self.Layer_Count[1]
        for i in range(diff):
            self.layers.insert(0,volly[len(volly)-1-i])
        self.Layer_Count=Layer_Count
            
    def removeLayer(self,x,Layer_Count):
        print("Layer(s) removed from Server")
        print(f"{self.Layer_Count[1]} => {Layer_Count[1]}")
        self.Layer_Count=Layer_Count