import torch
import torch.nn.functional as F

class External_attention(torch.nn.Module):
    '''
    Arguments:
        c (int): The input and output channel number.
    '''
    def __init__(self, c):
        super(External_attention, self).__init__()  
        self.conv1 = torch.nn.Conv1d(c, c, 1, bias=False) 

        self.k = 64
        self.linear_0 = torch.nn.Conv1d(c, self.k, 1, bias=False)
        self.linear_1 = torch.nn.Conv1d(self.k, c, 1, bias=False)

        self.conv2 = torch.nn.Conv1d(c, c, 1, bias=False) 

    def forward(self, x):
        idn = x
        x = self.conv1(x)
        attn = self.linear_0(x) # b, k, n
        attn = F.softmax(attn, dim=-1) # b, k, n

        attn = attn / (1e-9 + attn.sum(dim=1, keepdim=True)) #  # b, k, n
        x = self.linear_1(attn) # b, c, n

        x = self.conv2(x)
        x = x + idn
        x = F.relu(x)
        return x
    


class DenseLayer(torch.nn.Module):
    # model = DenseNet(layer_num=(6,12,24,16,16),
    #                      growth_rate=32, init_features=64,
    #                      in_channels=1 , middele_channels=128)
    def __init__(self,in_channels,middle_channels=128,out_channels=32):
        super(DenseLayer, self).__init__()
        self.layer = torch.nn.Sequential(
            torch.nn.BatchNorm1d(in_channels),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv1d(in_channels,middle_channels,3, padding=1),
            torch.nn.BatchNorm1d(middle_channels),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv1d(middle_channels,out_channels,5,padding=2)
        )
    def forward(self,x):
        return torch.cat([x, self.layer(x)],dim=1)


class DenseBlock(torch.nn.Sequential):
    def __init__(self,layer_num,growth_rate,in_channels,middele_channels=128):
        super(DenseBlock, self).__init__()
        for i in range(layer_num):
            layer = DenseLayer(in_channels+i*growth_rate,middele_channels,growth_rate)
            self.add_module('denselayer%d'%(i),layer)

class Transition(torch.nn.Sequential):
    def __init__(self,channels):
        super(Transition, self).__init__()
        self.add_module('norm',torch.nn.BatchNorm1d(channels))
        self.add_module('relu',torch.nn.ReLU(inplace=False))
        self.add_module('conv',torch.nn.Conv1d(channels,channels//2,3,padding=1))
        self.add_module('Avgpool',torch.nn.AvgPool1d(2))


class DenseNet(torch.nn.Module):
    def __init__(self,layer_num=(6,12,24,16),growth_rate=32,init_features=64,in_channels=1,middele_channels=128,classes=5):
        # Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0)
        super(DenseNet, self).__init__()
        self.feature_channel_num=init_features
        self.init_features = init_features
        self.conv = torch.nn.Conv1d(in_channels, self.feature_channel_num, kernel_size=7, stride=2,padding=3, bias=True)
        self.norm = torch.nn.BatchNorm1d(self.feature_channel_num)
        self.relu = torch.nn.ReLU()
        self.maxpool=torch.nn.MaxPool1d(3,2,1)

        self.DenseBlock1=DenseBlock(layer_num[0],growth_rate,self.feature_channel_num,middele_channels)
        self.feature_channel_num=self.feature_channel_num+layer_num[0]*growth_rate
        self.attn1 = External_attention(self.feature_channel_num)
        self.Transition1=Transition(self.feature_channel_num)

        self.DenseBlock2=DenseBlock(layer_num[1],growth_rate,self.feature_channel_num//2,middele_channels)
        self.feature_channel_num=self.feature_channel_num//2+layer_num[1]*growth_rate
        self.attn2 = External_attention(self.feature_channel_num)
        self.Transition2 = Transition(self.feature_channel_num)

        self.DenseBlock3 = DenseBlock(layer_num[2],growth_rate,self.feature_channel_num//2,middele_channels)
        self.feature_channel_num=self.feature_channel_num//2+layer_num[2]*growth_rate
        self.attn3 = External_attention(self.feature_channel_num)
        # self.Transition3 = Transition(self.feature_channel_num)

        # self.DenseBlock4 = DenseBlock(layer_num[3],growth_rate,self.feature_channel_num//2,middele_channels)
        # self.feature_channel_num=self.feature_channel_num//2+layer_num[3]*growth_rate
        # self.Transition4 = Transition(self.feature_channel_num)

        # self.DenseBlock5 = DenseBlock(layer_num[4],growth_rate,self.feature_channel_num//2,middele_channels)
        self.avgpool=torch.nn.AdaptiveAvgPool1d(1)

        self.output = torch.nn.Sequential(
            torch.nn.Conv1d(1174, 1174, 1), # 1324 = 1024 + 300(Lenth of the input)
            torch.nn.ReLU(),
            torch.nn.Conv1d(1174, 3000, 3, 1, 1), # 第二个数和输出长度一致。后同。
            torch.nn.ReLU(),
            torch.nn.Conv1d(3000, 3000, 1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(3000, 3000, 3, 1, 1)
        )


    def forward(self,x):
        bsz = x.size(0)
        # residual = x[:, 1, :].view(bsz, -1, 1)
        
        residual = x.view(bsz, -1, 1)
        x = x.view(bsz, 1, -1)
        
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.DenseBlock1(x)
        x = self.attn1(x)
        x = self.Transition1(x)

        x = self.DenseBlock2(x)
        x = self.attn2(x)
        x = self.Transition2(x)

        x = self.DenseBlock3(x)
        x = self.attn3(x)
        # x = self.Transition3(x)
        
        # x = self.DenseBlock4(x)
        # x = self.Transition4(x)

        # x = self.DenseBlock5(x)
        x = self.avgpool(x)

        x = torch.cat([x, residual], dim=1)
        x = self.output(x).view(bsz, -1)
        return x



if __name__ == '__main__':
    pass
    # input = torch.randn(size=(1,1,224))
    # model = DenseNet(layer_num=(6,12,24,16),growth_rate=32,in_channels=1,classes=5)
    # output = model(input)
    # print(output.shape)
    # print(model)
    # summary(model=model, input_size=(1, 224), device='cpu')