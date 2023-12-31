import torch.nn.functional as F
from torch import nn
import math
# Model at client side
import torch


class MobileNetV3Server(nn.Module):
    def __init__(self, config_name ="large",in_channels = 3,classes = 1000, ConvBlock=None,BNeck=None, Layers=None):
        super().__init__()
        config = self.config(config_name)
        self.Layer_Count=Layers.copy()
        # First convolution(conv2d) layer. 
        self.conv = ConvBlock(in_channels, 16, 3, 2, nn.Hardswish())
        # Bneck blocks in a list. 
        self.blocks = nn.ModuleList([])
        for c in config:
            kernel_size, exp_size, in_channels, out_channels, se, nl, s = c
            self.blocks.append(BNeck(in_channels, out_channels, kernel_size, exp_size, se, nl, s))
        
        # Classifier 
        last_outchannel = config[-1][3]
        last_exp = config[-1][1]
        out = 1280 if config_name == "large" else 1024
        self.classifier = nn.Sequential(
            ConvBlock(last_outchannel, last_exp, 1, 1, nn.Hardswish()),
            nn.AdaptiveAvgPool2d((1,1)),
            ConvBlock(last_exp, out, 1, 1, nn.Hardswish(), bn=False, bias=True),
            nn.Dropout(0.8),
            nn.Conv2d(out, classes, 1, 1)
        )
        print(self.state_dict().keys())
        self.layers=[]
        for i in range(16):
            if i>=self.Layer_Count[0]:
                self.layers.append(f"layer{i+1}")
            else:
                self.layers.append(None)
        print(self.layers)
        
    def _make_layers(self, in_channels, out_channels, block_cfg, width_multiplier):
        layers = []
        for cfg in block_cfg:
            expansion_factor, num_blocks, stride = cfg
            out_channels = int(out_channels * width_multiplier)
            for _ in range(num_blocks):
                layers.append(self.MobileNetV3Block(in_channels, out_channels, expansion_factor, stride))
                in_channels = out_channels
                stride = 1  # Only the first block in each stage has a stride > 1
        return nn.Sequential(*layers)
        
    
    def forward(self, x,Layer_Count, volly=None):
        if self.layers[2]:
           x =self.blocks[1](x)
        if self.layers[3]:
            x =self.blocks[2](x)
        if self.layers[4]:
            x =self.blocks[3](x)
        if self.layers[5]:
            x =self.blocks[4](x)
        if self.layers[6]:
            x =self.blocks[5](x)
        if self.layers[7]:
            x =self.blocks[6](x)
        if self.layers[8]:
            x =self.blocks[7](x)
        if self.layers[9]:
            x =self.blocks[8](x)
        if self.layers[10]:
            x =self.blocks[9](x)
        if self.layers[11]:
            x =self.blocks[10](x)
        if self.layers[12]:
            x =self.blocks[11](x)
        if self.layers[13]:
            x =self.blocks[12](x)
            
        x =self.blocks[13](x)
        x =self.blocks[14](x)

        x = self.classifier(x)
        y_hat =torch.flatten(x, 1)
        return y_hat
    
    def config(self, name):
        HE, RE = nn.Hardswish(), nn.ReLU()
        # [kernel, exp size, in_channels, out_channels, SEBlock(SE), activation function(NL), stride(s)] 
        large = [
                [3, 16, 16, 16, False, RE, 1],
                [3, 64, 16, 24, False, RE, 2],
                [3, 72, 24, 24, False, RE, 1],
                [5, 72, 24, 40, True, RE, 2],
                [5, 120, 40, 40, True, RE, 1],
                [5, 120, 40, 40, True, RE, 1],
                [3, 240, 40, 80, False, HE, 2],
                [3, 200, 80, 80, False, HE, 1],
                [3, 184, 80, 80, False, HE, 1],
                [3, 184, 80, 80, False, HE, 1],
                [3, 480, 80, 112, True, HE, 1],
                [3, 672, 112, 112, True, HE, 1],
                [5, 672, 112, 160, True, HE, 2],
                [5, 960, 160, 160, True, HE, 1],
                [5, 960, 160, 160, True, HE, 1]
        ]

        small = [
                [3, 16, 16, 16, True, RE, 2],
                [3, 72, 16, 24, False, RE, 2],
                [3, 88, 24, 24, False, RE, 1],
                [5, 96, 24, 40, True, HE, 2],
                [5, 240, 40, 40, True, HE, 1],
                [5, 240, 40, 40, True, HE, 1],
                [5, 120, 40, 48, True, HE, 1],
                [5, 144, 48, 48, True, HE, 1],
                [5, 288, 48, 96, True, HE, 2],
                [5, 576, 96, 96, True, HE, 1],
                [5, 576, 96, 96, True, HE, 1]
        ]

        if name == "large": return large
        if name == "small": return small
    
    def get_weights(self,client_dict,layers):
            keys=list(client_dict.keys())
            volly={}
            for i in range(len(keys)):
                for j in range(len(layers)):
                    if f"blocks.{layers[j]}." in keys[i]:
                        
                        volly[f"{keys[i]}"]=client_dict[keys[i]]                 
            self.Saved_Layers=volly.copy()
            return(volly)
        
    def activate_layers(self,layers):
        print("Server activate:",layers)
        all_layers=[]
        for i in range(3,17):
            all_layers.append(f"layer{i}")
        for i in range(len(layers)):
            self.layers[layers[i]-1]=all_layers[layers[i]-1]
            
        new_layer_count=0
        for i in range(len(self.layers)):
            if self.layers[i]!=None:
                new_layer_count=new_layer_count+1
        print(self.layers)
        self.Layer_Count=[16-new_layer_count,new_layer_count]
            
    def deactivate_layers(self,layers):
        print("Server Deactivate:",layers)
        for i in range(len(layers)):
            self.layers[layers[i]-1]=None
        new_layer_count=0
        for i in range(len(self.layers)):
            if self.layers[i]!=None:
                new_layer_count=new_layer_count+1
        print(self.layers)
        self.Layer_Count=[16-new_layer_count,new_layer_count]
        
        
        
        
        
        