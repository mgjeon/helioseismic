#==============================================================================
import torch
from torch import nn
from functools import partial
#==============================================================================
class Generator(nn.Module):
    def __init__(self, opt):
        super().__init__()
        #----------------------------------------------------------------------
        input_ch = opt.input_ch
        target_ch = opt.target_ch
        n_gf = opt.n_gf
        n_downsample = opt.n_downsample
        n_bottleneck = opt.n_bottleneck
        pad = get_pad_layer(opt.pad_type)
        norm = get_norm_layer(opt.norm_type)
        act = get_act_func(opt.act_type)
        mode = opt.mode
        #----------------------------------------------------------------------
        self.inp = nn.Sequential(
            pad(3),
            nn.Conv2d(input_ch, n_gf, kernel_size=7, padding=0),
            norm(n_gf),
            act(),
        )

        self.encoder = nn.ModuleList()
        dim_cur = n_gf
        for _ in range(n_downsample):
            dim_out = dim_cur * 2
            self.encoder.append(Down(dim_cur, dim_out, norm, act))
            dim_cur = dim_out

        self.bottleneck = nn.ModuleList()
        for _ in range(n_bottleneck):
            self.bottleneck.append(ResidualBlock(dim_cur, pad, norm, act))

        self.decoder = nn.ModuleList()
        for _ in range(n_downsample):
            dim_out = dim_cur // 2
            self.decoder.append(Up(dim_cur, dim_out, norm, act, mode=mode))
            dim_cur = dim_out

        self.out = nn.Sequential(
            pad(3),
            nn.Conv2d(dim_cur, target_ch, kernel_size=7, padding=0),
        )
        #----------------------------------------------------------------------

    def forward(self, x):
        #----------------------------------------------------------------------
        features = self.inp(x)
        en_features = []                         
        for block in self.encoder:
            en_features.append(features)
            features = block(features)
        
        for block in self.bottleneck:
            features = block(features)

        for block in self.decoder:
            features = block(features, en_features.pop())

        out = self.out(features)
        #----------------------------------------------------------------------
        return out


def get_pad_layer(pad_type):
    if pad_type == 'ReflectionPad2d':
        return nn.ReflectionPad2d
    if pad_type == 'ReplicationPad2d':
        return nn.ReplicationPad2d
    if pad_type == 'ZeroPad2d':
        return nn.ZeroPad2d
    raise NotImplementedError(f"Padding type [{pad_type}] is not valid")
    

def get_norm_layer(norm_type):
    if norm_type == 'BatchNorm2d':
        return partial(nn.BatchNorm2d, affine=True)
    if norm_type == 'InstanceNorm2d':
        return partial(nn.InstanceNorm2d, affine=False)
    raise NotImplementedError(f"Normalization type [{norm_type}] is not valid")


def get_act_func(act_type):
    if act_type == 'ReLU':
        return nn.ReLU
    if act_type == 'Mish':
        return nn.Mish
    if act_type == 'LeakyReLU':
        return partial(nn.LeakyReLU, negative_slope=0.2)
    raise NotImplementedError(f"Activation function type [{act_type}] is not valid")


class DoubleConv(nn.Module):
    """ (Conv => Norm => Act) * 2 """
    def __init__(self, inp_ch, out_ch, norm, act):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(inp_ch, out_ch, kernel_size=3, padding=1),
            norm(out_ch),
            act(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            norm(out_ch),
            act()
        )

    def forward(self, x):
        return self.double_conv(x)
    

class Down(nn.Module):
    """ Downscaling => DoubleConv """
    def __init__(self, inp_ch, out_ch, norm, act):
        super().__init__()
        self.down = nn.MaxPool2d(2)
        self.conv = DoubleConv(inp_ch, out_ch, norm, act)

    def forward(self, x):
        x = self.down(x)
        return self.conv(x)
    

class Up(nn.Module):
    """ Upscaling => SkipConnection => DoubleConv """
    def __init__(self, inp_ch, out_ch, norm, act, mode='nearest'):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode=mode)
        self.conv = DoubleConv(inp_ch + (inp_ch // 2), out_ch, norm, act)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat((x2, x1), dim=1)
        return self.conv(x)
    

class ResidualBlock(nn.Module):
    def __init__(self, n_ch, pad, norm, act):
        super().__init__()
        self.res_block = nn.Sequential(
            pad(1),
            nn.Conv2d(n_ch, n_ch, kernel_size=3, padding=0, stride=1),
            norm(n_ch),
            act(),

            pad(1),
            nn.Conv2d(n_ch, n_ch, kernel_size=3, padding=0, stride=1),
            norm(n_ch)
        )

    def forward(self, x):
        return x + self.res_block(x)


#==============================================================================
class Discriminator(nn.Module):
    def __init__(self, opt):
        super().__init__()
        #----------------------------------------------------------------------
        self.n_D = opt.n_D
        #----------------------------------------------------------------------
        for i in range(self.n_D):
            setattr(self, f'scale_{i}', PatchDiscriminator(opt))
        #----------------------------------------------------------------------

    def forward(self, x):
        result = []
        for i in range(self.n_D):
            block = getattr(self, f'scale_{i}')
            result.append(block(x))
            if i != (self.n_D - 1):
                x = nn.AvgPool2d(kernel_size=3, stride=2, padding=1, count_include_pad=False)(x)
                
        return result
    

class PatchDiscriminator(nn.Module):
    def __init__(self, opt):
        super().__init__()
        #----------------------------------------------------------------------
        input_channel = opt.input_ch + opt.target_ch
        n_df = opt.n_df
        # norm = get_norm_layer(opt.norm_type)
        # act = get_act_func(opt.act_type)
        norm = nn.InstanceNorm2d
        act = partial(nn.LeakyReLU, negative_slope=0.2)
        #----------------------------------------------------------------------
        blocks = []
        blocks += [[nn.Conv2d(input_channel, n_df  , kernel_size=4, stride=2, padding=1), act()]]
        blocks += [[nn.Conv2d(n_df         , n_df*2, kernel_size=4, stride=2, padding=1), norm(n_df*2), act()]]
        blocks += [[nn.Conv2d(n_df*2       , n_df*4, kernel_size=4, stride=2, padding=1), norm(n_df*4), act()]]
        blocks += [[nn.Conv2d(n_df*4       , n_df*8, kernel_size=4, stride=1, padding=1), norm(n_df*8), act()]]
        blocks += [[nn.Conv2d(n_df*8       , 1     , kernel_size=4, stride=1, padding=1)]] 

        self.n_blocks = len(blocks)
        for i in range(self.n_blocks):
            setattr(self, f'block_{i}', nn.Sequential(*blocks[i]))
        #----------------------------------------------------------------------
    
    def forward(self, x):
        result = [x]
        for i in range(self.n_blocks):
            block = getattr(self, f'block_{i}')
            result.append(block(result[-1]))

        return result[1:]

#==============================================================================
class Loss:
    def __init__(self, opt, device):
        super().__init__()
        # self.opt = opt
        # self.device = torch.device('cuda:0' if opt.gpu_ids != -1 else 'cpu:0')
        self.device = device

        self.L1 = nn.L1Loss()
        self.L2 = nn.MSELoss()
        self.n_D = opt.n_D
        self.n_CC = opt.n_CC
        self.ccc = opt.ccc
        self.eps = opt.eps
        self.lambda_LSGAN = opt.lambda_LSGAN
        self.lambda_FM = opt.lambda_FM
        self.lambda_CC = opt.lambda_CC

    def __call__(self, D, G, inputs, target):
        loss_D = 0
        loss_G = 0
        loss_G_LSGAN = 0
        loss_G_FM = 0
        loss_G_CC = 0
        #------------------------------------------------------------------------------
        fake = G(inputs)

        #------------------------------------------------------------------------------
        # LSGAN loss for Discriminator
        real_features = D(torch.cat((inputs, target), dim=1))
        fake_features = D(torch.cat((inputs, fake.detach()), dim=1))
        for i in range(self.n_D):
            ones_grid = get_grid(real_features[i][-1], is_real=True).to(self.device)
            zero_grid = get_grid(fake_features[i][-1], is_real=False).to(self.device)

            loss_D += (
                self.L2(real_features[i][-1], ones_grid) +   # D(x, y) -> 1
                self.L2(fake_features[i][-1], zero_grid)     # D(x, G(x)) -> 0 
            )*0.5
        #------------------------------------------------------------------------------
        # LSGAN loss & FM loss for Generator
        fake_features = D(torch.cat((inputs, fake), dim=1))
        for i in range(self.n_D):
            ones_grid = get_grid(fake_features[i][-1], is_real=True).to(self.device)
            loss_G_LSGAN += self.L2(fake_features[i][-1], ones_grid) * 0.5
            
            for j in range(len(fake_features[i])):
                loss_G_FM += self.L1(fake_features[i][j], real_features[i][j].detach()) * (1.0 / self.n_D)

        #------------------------------------------------------------------------------
        # CC loss for Generator
        for i in range(self.n_CC):
            real_down = target.to(self.device)
            fake_down = fake.to(self.device)
            for _ in range(i):
                real_down = nn.AvgPool2d(kernel_size=3, stride=2, padding=1, count_include_pad=False)(real_down)
                fake_down = nn.AvgPool2d(kernel_size=3, stride=2, padding=1, count_include_pad=False)(fake_down)
            
            loss_G_CC += self.__Inspector(real_down, fake_down) * (1.0 / self.n_CC)

        #------------------------------------------------------------------------------
        loss_G = (
            self.lambda_LSGAN * loss_G_LSGAN +
            self.lambda_FM * loss_G_FM +
            self.lambda_CC * loss_G_CC
        )
        return loss_D, loss_G, loss_G_LSGAN, loss_G_FM, loss_G_CC
    
        
    def __Inspector(self, real, fake):
        rd = real - torch.mean(real)
        fd = fake - torch.mean(fake)

        r_num = torch.sum(rd * fd)
        r_den = torch.sqrt(torch.sum(rd**2)) * torch.sqrt(torch.sum(fd**2))
        pcc_val = r_num / (r_den + self.eps)

        if self.ccc:
            num = 2 * pcc_val * torch.std(real) * torch.std(fake)
            den = torch.var(real) + torch.var(fake) + (torch.mean(real) - torch.mean(fake))**2
            ccc_val = num / (den + self.eps)
            return 1 - ccc_val
        else:
            return 1 - pcc_val


def get_grid(inputs, is_real=True):
    if is_real:
        grid = torch.ones_like(inputs)

    elif not is_real:
        grid = torch.zeros_like(inputs)

    return grid

#=============================================================================
def weights_init(module):
    if isinstance(module, nn.Conv2d):
        module.weight.detach().normal_(0.0, 0.02)
    elif isinstance(module, nn.BatchNorm2d):
        module.weight.detach().normal_(1.0, 0.02)
        module.bias.detach().fill_(0.0)