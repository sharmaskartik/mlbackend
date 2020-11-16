import torch.nn as nn
import torch.nn.functional as F
import torch


class p300Model(nn.Module):
    def __init__(self, input_channels, init_channels, n_classes):
        super(p300Model, self).__init__()
        #first convolution applied temporally
        self.conv1 = nn.Conv1d(1, init_channels, 10, stride = 1)

        #second convolution is applied accross depth
        self.conv2 = nn.Conv2d(input_channels, init_channels, (init_channels, 1), stride = 1)
        self.batch_norm_1 = nn.BatchNorm1d(init_channels)
        self.pool1 = nn.MaxPool1d(3, 3)

        self.conv_pool2 = ConvPool(init_channels, 2 * init_channels, 10, 3)
        self.conv_pool3 = ConvPool(2 * init_channels, 4 * init_channels, 10, 3)
        self.conv_pool4 = ConvPool(4 * init_channels, 8 * init_channels, 10, 3)
        #self.avg_pool = nn.AvgPool2d(10, (8 * init_channels ,8))
        self.avg_pool = nn.AvgPool2d(2, (8 * init_channels, 1))

    def forward(self, x):
        #Apply the temporal convolution (in the first layer)
        output = []
        for i in range(x.size()[1]):
            res = self.conv1(x[:, i:i+1, :])
            output.append(res.unsqueeze(1))
        x = torch.cat(output, dim=1)

        x = F.elu(self.batch_norm_1(self.conv2(x).squeeze(2)))
        #x = self.pool1(x)

        x = self.conv_pool2(x)
        x = self.conv_pool3(x)
        x = self.conv_pool4(x)

        x = self.avg_pool(x).squeeze(1)
        return x


class ConvPool(nn.Module):
    def __init__(self, input_channels, out_channels, conv_kernel_size,
                    pooling_kernel_size, conv_stride = 1,  pooling_stride = 3):
        super(ConvPool, self).__init__()
        self.conv = nn.Conv1d(input_channels, out_channels, conv_kernel_size, stride = conv_stride)
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.pool = nn.MaxPool1d(pooling_kernel_size, pooling_stride)

    def forward(self, x):
        x = F.elu(self.batch_norm(self.conv(x)))
        x = self.pool(x)

        return x
