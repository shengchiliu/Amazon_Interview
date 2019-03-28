import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F


class ChannelledVAE(nn.Module):
    def __init__(self, dim_z, dim_input):
        super(ChannelledVAE, self).__init__()
        self.dim_z = dim_z
        self.h, self.w, self.d = dim_input
        self.size = self.h * self.w * self.d

        self.e1 = nn.Linear(self.size, 512)    # (128x128x2), 512
        self.e2 = nn.Linear(512,       128)
        self.e3 = nn.Linear(128,        32)
        self.e4 = nn.Linear(32,      dim_z)
        self.e5 = nn.Linear(32,      dim_z)
        self.d4 = nn.Linear(dim_z,      32)
        self.d3 = nn.Linear(32,        128)
        self.d2 = nn.Linear(128,       512)
        self.d1 = nn.Linear(512, self.size)    # 512, (128x128x2)

        self.relu = nn.ReLU()
        self.lklu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.weight_init()

    def encode(self, x):
        xe  = self.relu(self.e1(x))                         #; print(xe.shape)
        xe  = self.lklu(self.e2(xe))                        #; print(xe.shape)        
        xe  = self.lklu(self.e3(xe))                        #; print(xe.shape)
        xe0 = self.e4(xe)                                   #; print(xe0.shape)
        xe1 = self.e5(xe)                                   #; print(xe1.shape)
        c = xe0[:,0:1].add_(xe1[:,0:1])                     #; print(c.shape)
        c = self.tanh(c)                                    #; print(c.shape)
        mu = xe0[:,1:]                                      #; print(mu.shape)
        logvar = xe1[:,1:]                                  #; print(logvar.shape)
        return c, mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)                     #; print(logvar.shape); print(std.shape)
            eps = torch.randn_like(std)                     #; print(eps.shape); print(mu.shape)
            return eps.mul(std).add_(mu)
        else:
            return mu
        
    def decode(self, c, z):
        xd = torch.cat((c, z), 1)                           #; print('cat(c,z)'); print(xd.shape)
        xd = self.lklu(self.d4(xd))                         #; print(xd.shape)
        xd = self.lklu(self.d3(xd))                         #; print(xd.shape)
        xd = self.lklu(self.d2(xd))                         #; print(xd.shape)
        xd = self.sigmoid(self.d1(xd))                      #; print(xd.shape)
        return xd

    def weight_init(self):
        for block in self._modules:
            kaiming_init(self._modules[block])

    def forward(self, x):
        c, mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        decoded = self.decode(c, z)
        return decoded, c, mu, logvar, z


class BetaVAE(nn.Module):
    def __init__(self, dim_z, dim_input):
        super(BetaVAE, self).__init__()
        self.dim_z = dim_z
        self.size = dim_input

        self.e1 = nn.Linear(self.size, 512)    # (128x128x2), 512
        self.e2 = nn.Linear(512,       128)
        self.e3 = nn.Linear(128,        32)
        self.e4 = nn.Linear(32,      dim_z)
        self.e5 = nn.Linear(32,      dim_z)
        self.d4 = nn.Linear(dim_z,      32)
        self.d3 = nn.Linear(32,        128)
        self.d2 = nn.Linear(128,       512)
        self.d1 = nn.Linear(512, self.size)    # 512, (128x128x2)

        self.relu = nn.ReLU()
        self.lklu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.weight_init()

    def encode(self, x):
        xe = self.relu(self.e1(x))                          #; print(xe.shape)
        xe = self.lklu(self.e2(xe))                         #; print(xe.shape)        
        xe = self.lklu(self.e3(xe))                         #; print(xe.shape)
        mu = self.e4(xe)                                    #; print(mu.shape)
        logvar = self.e5(xe)                                #; print(logvar.shape)        
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)                     #; print(logvar.shape); print(std.shape)
            eps = torch.randn_like(std)                     #; print(eps.shape); print(mu.shape)
            return eps.mul(std).add_(mu)
        else:
            return mu
        
    def decode(self, z):
        xd = self.lklu(self.d4(z))                          #; print(xd.shape)
        xd = self.lklu(self.d3(xd))                         #; print(xd.shape)
        xd = self.lklu(self.d2(xd))                         #; print(xd.shape)
        xd = self.sigmoid(self.d1(xd))                      #; print(xd.shape)
        return xd

    def weight_init(self):
        for block in self._modules:
            kaiming_init(self._modules[block])

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        decoded = self.decode(z)
        return decoded, mu, logvar, z


class AE(nn.Module):
    def __init__(self, dim_z, dim_input):
        super(AE, self).__init__()
        self.dim_z = dim_z
        self.h, self.w, self.d = dim_input
        self.size = self.h * self.w * self.d

        self.e1 = nn.Linear(self.size, 512)    # (128x128x2), 512
        self.e2 = nn.Linear(512,       128)
        self.e3 = nn.Linear(128,        32)
        self.e4 = nn.Linear(32,      dim_z)
        self.d4 = nn.Linear(dim_z,      32)
        self.d3 = nn.Linear(32,        128)
        self.d2 = nn.Linear(128,       512)
        self.d1 = nn.Linear(512, self.size)    # 512, (128x128x2)

        self.relu = nn.ReLU()
        self.lklu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.weight_init()

    def encode(self, x):
        xe = self.relu(self.e1(x))                          #; print(xe.shape)
        xe = self.lklu(self.e2(xe))                         #; print(xe.shape)
        xe = self.lklu(self.e3(xe))                         #; print(xe.shape)
        xe = self.e4(xe)                                    #; print(mu.shape)
        c = self.sigmoid(xe[:,0:1].add_(xe[:,0:1]))         #; print(c.shape)
        return c, xe, xe
        
    def decode(self, c, z):
        xd = self.lklu(self.d4(z))                          #; print(xd.shape)
        xd = self.lklu(self.d3(xd))                         #; print(xd.shape)
        xd = self.lklu(self.d2(xd))                         #; print(xd.shape)
        xd = self.sigmoid(self.d1(xd))                      #; print(xd.shape)
        return xd

    def weight_init(self):
        for block in self._modules:
            kaiming_init(self._modules[block])

    def forward(self, x):
        c, z, logvar = self.encode(x)
        decoded = self.decode(c, z)
        return decoded, c, z, logvar, z


class DropoutBVAE(nn.Module):
    def __init__(self, dim_z, dim_input, device):
        super(DropoutBVAE, self).__init__()
        self.dim_z = dim_z
        self.h, self.w, self.d = dim_input
        self.size = self.h * self.w * self.d
        self.device = device

        self.e1 = nn.Linear(self.size, 512)     # (128x128x2), 512
        self.e2 = nn.Linear(512,       128)
        self.e3 = nn.Linear(128,        32)
        self.e4 = nn.Linear(32,      dim_z)
        self.e5 = nn.Linear(32,      dim_z)
        self.d4 = nn.Linear(dim_z,      32)
        self.d3 = nn.Linear(32,        128)
        self.d2 = nn.Linear(128,       512)
        self.d1 = nn.Linear(512, self.size)     # 512, (128x128x2)

        self.relu = nn.ReLU()
        self.lklu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.weight_init()

    def encode(self, x):
        xe = self.relu(self.e1(x))                          #; print(xe.shape)
        xe = self.lklu(self.e2(xe))                         #; print(xe.shape)        
        xe = self.lklu(self.e3(xe))                         #; print(xe.shape)
        mu = self.e4(xe)                                    #; print(mu.shape)
        logvar = self.e5(xe)                                #; print(logvar.shape)        
        c = self.sigmoid(mu[:,0:1].add_(logvar[:,0:1]))     #; print(c.shape)
        return c, mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)                     #; print(logvar.shape); print(std.shape)
            eps = torch.randn_like(std)                     #; print(eps.shape); print(mu.shape)
            return eps.mul(std).add_(mu)
        else:
            return mu
        
    def decode(self, c, z):
        z1 = torch.zeros(z.shape).to(self.device)           # z1 = self.dropout(z)
        z1[:, 7] = z[:, 7]
        xd = self.lklu(self.d4(z1))                         #; print(xd.shape)
        xd = self.lklu(self.d3(xd))                         #; print(xd.shape)
        xd = self.lklu(self.d2(xd))                         #; print(xd.shape)
        xd = self.sigmoid(self.d1(xd))                      #; print(xd.shape)
        return xd

    def weight_init(self):
        for block in self._modules:
            kaiming_init(self._modules[block])

    def forward(self, x):
        c, mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        decoded = self.decode(c, z)
        return decoded, c, mu, logvar, z


class DropoutCVAE(nn.Module):
    def __init__(self, dim_z, dim_input, device):
        super(DropoutCVAE, self).__init__()

        # Keep for reference
        self.dim_z = dim_z
        self.h, self.w, self.d = dim_input
        self.size = self.h * self.w * self.d
        self.device = device

        # Define layers
        self.e1 = nn.Linear(self.size, 512)     # (128x128x2), 512
        self.e2 = nn.Linear(512,       128)
        self.e3 = nn.Linear(128,        32)
        self.e4 = nn.Linear(32,      dim_z)
        self.e5 = nn.Linear(32,      dim_z)
        self.d4 = nn.Linear(dim_z,      32)
        self.d3 = nn.Linear(32,        128)
        self.d2 = nn.Linear(128,       512)
        self.d1 = nn.Linear(512, self.size)     # 512, (128x128x2)

        # Define Activation Functions
        self.relu = nn.ReLU()
        self.lklu = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.dropout = nn.Dropout(p=0.5)
        self.weight_init()

    def encode(self, x):
        xe  = self.relu(self.e1(x))                         #; print(xe.shape)
        xe  = self.lklu(self.e2(xe))                        #; print(xe.shape)        
        xe  = self.lklu(self.e3(xe))                        #; print(xe.shape)
        xe0 = self.e4(xe)                                   #; print(xe0.shape)
        xe1 = self.e5(xe)                                   #; print(xe1.shape)
        c = xe0[:,0:1].add_(xe1[:,0:1])                     #; print(c.shape)
        c = self.tanh(c)                                    #; print(c.shape)
        mu = xe0[:,1:]                                      #; print(mu.shape)
        logvar = xe1[:,1:]                                  #; print(logvar.shape)
        return c, mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)                     #; print(logvar.shape); print(std.shape)
            eps = torch.randn_like(std)                     #; print(eps.shape); print(mu.shape)
            return eps.mul(std).add_(mu)
        else:
            return mu
        
    def decode(self, c, z):
        if self.training:
            z1 = torch.zeros(z.shape).to(self.device)       # z1 = self.dropout(z)
        else:
            z1 = torch.zeros(z.shape).cuda()                # z1 = self.dropout(z)
        z1[:, 7] = 2*z[:, 7]
        # if self.training:
        #     z1[:, 7] = 2*z[:, 7]                          # double the weight, dropout process
        # else:
        #     z1 = z
        xd = torch.cat((c, z1), 1)                          #; print('cat(c,z)'); print(xd.shape)
        xd = self.lklu(self.d4(xd))                         #; print(xd.shape)
        xd = self.lklu(self.d3(xd))                         #; print(xd.shape)
        xd = self.lklu(self.d2(xd))                         #; print(xd.shape)
        xd = self.sigmoid(self.d1(xd))                      #; print(xd.shape)
        return xd

    def weight_init(self):
        for block in self._modules:
            kaiming_init(self._modules[block])

    def forward(self, x):
        c, mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        decoded = self.decode(c, z)
        return decoded, c, mu, logvar, z


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()


if __name__ == "__main__":
    dim_z = 16
    dim_input = [128, 128, 2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # exp_A if ture else exp_B
    x = torch.randn(10, 128*128*2).to(device)
    print('x: {}\n'.format(x.shape))
    # model = AE(dim_z, dim_input)
    # model = BetaVAE(dim_z, dim_input)
    # model = ChannelledVAE(dim_z, dim_input)
    # model = DropoutBVAE(dim_z, dim_input, device).to(device)
    model = DropoutCVAE(dim_z, dim_input, device).to(device)
    y, c, mu, logvar, z = model(x)
    print('\ny: {}\nc: {}\nmu: {}\nlogvar: {}\nz: {}\n'.format(y.shape, c.shape, mu.shape, logvar.shape, z.shape))

