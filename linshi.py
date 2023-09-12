class SPPX(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SPPX, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            Conv(in_channel, out_channel, 1, act=False),
        )
        self.branch1 = nn.Sequential(
            Conv(in_channel, out_channel, 1, act=False),
            Conv(out_channel, out_channel, k=(1, 3), p=(0, 1), act=False),
            Conv(out_channel, out_channel, k=(3, 1), p=(1, 0), act=False),
            Conv(out_channel, out_channel, 3, p=3, d=3, act=False)
        )
        self.branch2 = nn.Sequential(
            Conv(in_channel, out_channel, 1, act=False),
            Conv(out_channel, out_channel, k=(1, 5), p=(0, 2), act=False),
            Conv(out_channel, out_channel, k=(5, 1), p=(2, 0), act=False),
            Conv(out_channel, out_channel, 3, p=5, d=5, act=False)
        )
        self.branch3 = nn.Sequential(
            Conv(in_channel, out_channel, 1),
            Conv(out_channel, out_channel, k=(1, 7), p=(0, 3), act=False),
            Conv(out_channel, out_channel, k=(7, 1), p=(3, 0), act=False),
            Conv(out_channel, out_channel, 3, p=7, d=7)
        )
        self.conv_cat = Conv(4*out_channel, out_channel, 3, p=1, act=False)
        self.conv_res = Conv(in_channel, out_channel, 1, act=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))

        print(x.shape)
    
        return x 
