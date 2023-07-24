import torch.nn.functional as F
from torch import nn
import math

# Model at server side
class ResNet18_server_side(nn.Module):
    def __init__(self,global_server, block, num_layers, classes,channels):
        super(ResNet18_server_side, self).__init__()
        self.Global=global_server
        self.Layer_Count=[2,4]
        self.input_planes = 64
        self.Saved_Layers={}
        
        #OG split
        self.layer3 = nn.Sequential (
                nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1),
                nn.BatchNorm2d(64),
                nn.ReLU (inplace = True),
                nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1),
                nn.BatchNorm2d(64),       
                )

        self.layer4 = self._layer(block, 128, num_layers[1], stride = 2)
        self.layer5 = self._layer(block, 256, num_layers[2], stride = 2)
        self.layer6 = self._layer(block, 512, num_layers[3], stride = 2)
        
        self.layers=[]
        for i in range(6):
            if i>=self.Layer_Count[0]:
                self.layers.append(f"layer{i+1}")
            else:
                self.layers.append(None)
        print(self.layers)
        
        self. averagePool = nn.AvgPool2d(kernel_size = 7, stride = 1)
        self.fc = nn.Linear(512 * block.expansion, classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
        
    def _layer(self, block, planes, num_layers, stride = 2):
        dim_change = None
        if stride != 1 or planes != self.input_planes * block.expansion:
            dim_change = nn.Sequential(nn.Conv2d(self.input_planes, planes*block.expansion, kernel_size = 1, stride = stride),
                                       nn.BatchNorm2d(planes*block.expansion))
        netLayers = []
        netLayers.append(block(self.input_planes, planes, stride = stride, dim_change = dim_change))
        self.input_planes = planes * block.expansion
        for i in range(1, num_layers):
            netLayers.append(block(self.input_planes, planes))
            self.input_planes = planes * block.expansion
            
        return nn.Sequential(*netLayers)
    
    
    def forward(self, x,Layer_Count,volly=None): 
        if self.layers[2]!=None:
            out2 = self.layer3(x)
            out2 = out2 + x          # adding the resudial inputs -- downsampling not required in this layer
            x = F.relu(out2)
        if self.layers[3]:
            x = self.layer4(x)
        if self.layers[4]:
            x = self.layer5(x)
        x6 = self.layer6(x)
        x7 = F.avg_pool2d(x6, 2)
        x8 = x7.view(x7.size(0), -1) 
        y_hat =self.fc(x8)
        return y_hat
    
        
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