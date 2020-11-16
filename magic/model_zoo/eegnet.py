import torch.nn as nn
import torch.nn.functional as F
import torch


class EEGNet(nn.Module):
    def __init__(self, C, F1, D, kernel_size = (1,128), drop_p=0.5):
        super(EEGNet, self).__init__()

        F2 = D * F1

        self.conv2d_1 = nn.Conv2d(1, F1, kernel_size, stride=1, padding=(0, int(kernel_size[1]/2)))
        self.batch_norm_1 = nn.BatchNorm2d(F1)
        self.depthwise_conv = nn.Conv2d(F1, F2, (C, 1), stride=1)
        self.batch_norm_2 = nn.BatchNorm2d(F2)
        self.avg_pool_1 = nn.AvgPool2d((1,4))
        self.dropout_1 = nn.Dropout2d(drop_p)


        self.conv2d_2 = nn.Conv2d(F2, F2, (1, 16), stride=1, padding=(0, 8))
        self.conv2d_3 = nn.Conv2d(F2, F2, (1, 1), stride=1)
        self.batch_norm_3 = nn.BatchNorm2d(F2)
        self.avg_pool_2 = nn.AvgPool2d((1, 8))
        self.dropout_2 = nn.Dropout2d(drop_p)

        self.final_pool = nn.AvgPool2d(4,3)
        self.final_pool = nn.AvgPool2d(3,7)

    def forward(self, x):
        x = self.conv2d_1(x)
        x = self.batch_norm_1(x)
        x = self.depthwise_conv(x)
        x = self.batch_norm_2(x)
        x = F.elu(x)
        x = self.avg_pool_1(x)
        x = self.dropout_1(x)

        x = self.conv2d_2(x)
        x = self.conv2d_3(x)
        x = self.batch_norm_3(x)
        x = F.elu(x)
        x = self.avg_pool_2(x)
        x = self.dropout_2(x)

        x = self.final_pool(x.squeeze(2)).squeeze(2)

        return x
