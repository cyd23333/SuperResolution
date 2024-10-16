import torch.nn as nn
        
class SRPCNN(nn.Module):
    def __init__(self, alpha, sw):
        super(SRPCNN, self).__init__()
        self.alpha = alpha
        self.sw = sw
        
    # (Filter_Size, Num_of_Filters, Num_of_Feature_Vecs): (9,256,1) x1, (5,256,256) x7 Layers, (7,1,256) x1
    # Reminder: PReLU(x) = max(0,x) + a * min(0,x)
    # Reminder: Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
    # Reminder: ConvTranspose1d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1)
        if sw == 1:
            preluLayerType01 = nn.PReLU(num_parameters=1, init=0.25)
            preluLayerType02 = nn.PReLU(num_parameters=256, init=0.25)
            preluLayerType03 = nn.PReLU(num_parameters=1, init=0.25)
            self.conv1 = nn.Sequential(nn.Conv1d(1, 256, 9), preluLayerType01)
            self.conv2 = nn.ModuleList([nn.Sequential(nn.Conv1d(256, 256, 5), preluLayerType02) for _ in range(7)])
            self.deconv1 = nn.Sequential(nn.ConvTranspose1d(256, 1, 7, stride=alpha), preluLayerType03)
            self.deconv2 = nn.ConvTranspose1d(1, 1, 734)
        else:
            pass

    def forward(self, x):
        sw = self.sw
        x = x.view(x.size(0), 1, x.size(1))
        
        if sw == 1:
            x = self.conv1(x)
            for layer in self.conv2:
                x = layer(x)
            x = self.deconv1(x)
            x = self.deconv2(x)
        else:
            pass
        
        x = x.view(x.size(0), x.size(2))
        return x