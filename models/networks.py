import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import models
from . import deeplab

###############################################################################
# Functions
###############################################################################
def weights_init_normal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.uniform(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.uniform(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

def weights_init_xavier_U(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('ConvTranspose2d') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=1)
        
def weights_init_xavier_D(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv2d') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=1)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'xavier_U':
        net.apply(weights_init_xavier_U)
    elif init_type == 'xavier_D':
        net.apply(weights_init_xavier_D)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        if opt.which_model_netG == 'deeplab_aspp':
            def lambda_rule(epoch):
                lr_l = (max(0.001, 1.0 - epoch/30.0)) ** 0.9
                return lr_l
        else:
            def lambda_rule(epoch):
                lr_l = 0.1 ** max(0.0, epoch//14.0) #for LIP
                return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def define_G(input_nc, output_nc, ngf, which_model_netG, hook, input_size, norm='batch', use_dropout=False, init_type='normal', gpu_ids=[]):
    netG = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())

    if which_model_netG == 'resnet_9blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, gpu_ids=gpu_ids)
    elif which_model_netG == 'resnet_6blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6, gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_128':
        netG = UnetGenerator(input_nc, output_nc, 7, hook, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_256':
        netG = UnetGenerator(input_nc, output_nc, 8, hook, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
        init_weights(netG, init_type = 'xavier_U')
    elif which_model_netG == 'deeplab_aspp':
        netG = deeplab.D_ResNet(deeplab.D_Bottleneck, [3, 4, 23, 3], output_nc, input_size)
        init_weights(netG, init_type = 'xavier')
        model_res101 = models.resnet101(pretrained=True)
        model_res101 = model_res101.cuda()
        pretrained_dict = model_res101.state_dict()
        new_params = netG.state_dict().copy()
        for i in new_params:
            i_parts = i.split('.')
            if i_parts[0] != 'layer5':
                new_params['.'.join(i_parts)] = pretrained_dict[i]
        netG.load_state_dict(new_params)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    if len(gpu_ids) > 0:
        netG.cuda(gpu_ids[0])
    return netG


def define_D(input_nc, ndf, which_model_netD,
             n_layers_D=3, norm='batch', use_sigmoid=False, init_type='normal', gpu_ids=[]):
    netD = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())
    if which_model_netD == 'basic':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    elif which_model_netD == 'n_layers' and n_layers_D < 5:
        netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                  which_model_netD)
    if use_gpu:
        netD.cuda(gpu_ids[0])
    init_weights(netD, init_type=init_type)
    return netD


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


##############################################################################
# Classes
##############################################################################

# Flatten the tensor for fc layer
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
    
# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()
            
    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, gpu_ids=[], padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)

# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class Bottleneck(nn.Module):
    def __init__(self, model_cx, model_x):
        super(Bottleneck, self).__init__()
        self.ReLU = nn.ReLU()
        self.model_cx = self.build_block(model_cx)
        self.model_x = self.build_block(model_x)
        
    def build_block(self, model):
        return nn.Sequential(*model)
    
    def forward(self, x):
        x = self.ReLU(x)
        if len(self.model_x) == 0:
            return self.model_cx(x) + x
        else:
            return self.model_cx(x) + self.model_x(x)

class ASPP_Module(nn.Module):
    def __init__(self, input_nc, conv2d_list):
        super(ASPP_Module, self).__init__()
        self.conv2d_list = conv2d_list
        self.conv1_1 = nn.Conv2d(input_nc * 4, input_nc, kernel_size = 1)
        self.conv1_1.weight.data.normal_(0, 0.01)
        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)
        
    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list)-1):
            out = torch.cat([out, self.conv2d_list[i+1](x)], 1)
        out = self.conv1_1(out)
        return out

class UnetHook():
    def __init__(self):
        self.value = 0
    
    def hook_out(self, module, input, output):
        self.value = output
        
    def get_value(self):
        return self.value
    
    def print_value(self):
        print(self.value)
    
# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, hook, ngf=64, 
                 norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(UnetGenerator, self).__init__()
        self.gpu_ids = gpu_ids
        
        #Resnet 101
        model_res101 = models.resnet101(pretrained=True)
        model_res101 = model_res101.cuda()
        
        # construct unet structure
        T_block = UnetSkipConnectionBlock(output_nc, ngf * 32, input_nc = ngf * 32, submodule = None, depth = -2, norm_layer = norm_layer, model_ft = model_res101)
        handle = T_block.register_forward_hook(hook.hook_out)
        U_block = UnetSkipConnectionBlock(output_nc, ngf * 32, input_nc = None, submodule = T_block, depth = -1, norm_layer = norm_layer, model_ft = model_res101) 

        U_block = UnetSkipConnectionBlock(ngf * 16, ngf * 32, input_nc = None, submodule = U_block, depth = 0, norm_layer = norm_layer, model_ft = model_res101)
        U_block = UnetSkipConnectionBlock(ngf * 8, ngf * 16, input_nc = None, submodule = U_block, depth = 1, norm_layer = norm_layer, model_ft = model_res101)
        U_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc = None, submodule = U_block, depth = 2, norm_layer = norm_layer, model_ft = model_res101)
        U_block = UnetSkipConnectionBlock(ngf, ngf * 4, input_nc = None, submodule = U_block, depth = 3, norm_layer = norm_layer, model_ft = model_res101)
        U_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc = input_nc, submodule = U_block, depth = 4, norm_layer = norm_layer, model_ft = model_res101)

        self.model = U_block

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, depth, input_nc=None,
                 submodule=None,  norm_layer=nn.BatchNorm2d, use_dropout=False, model_ft=None):
        super(UnetSkipConnectionBlock, self).__init__()
        
        assert(depth <= 4)
        self.depth = depth
        
        #======================== depth 4 ==========================
        ResBlock0 = [model_ft.conv1, model_ft.bn1]
    
        #======================== depth 3 ==========================
        ResBlock1 = [model_ft.maxpool,]
        for i in range(3):
            model_x = []
            model_cx = []
            if i == 0:
                model_x = [model_ft.layer1[i].downsample[0],
                           model_ft.layer1[i].downsample[1]]
            model_cx = [model_ft.layer1[i].conv1, 
                        model_ft.layer1[i].bn1,
                        model_ft.layer1[i].conv2,
                        model_ft.layer1[i].bn2,
                        model_ft.layer1[i].conv3,
                        model_ft.layer1[i].bn3]
            
            ResBlock1 += [Bottleneck(model_cx, model_x),]
        
        #======================== depth 2 ==========================
        ResBlock2 = []
        for j in range(4):
            model_x = []
            model_cx = []
            if j == 0:
                model_x = [model_ft.layer2[j].downsample[0], 
                           model_ft.layer2[j].downsample[1]]
            model_cx = [model_ft.layer2[j].conv1, 
                     model_ft.layer2[j].bn1,
                     model_ft.layer2[j].conv2,
                     model_ft.layer2[j].bn2,
                     model_ft.layer2[j].conv3,
                     model_ft.layer2[j].bn3]
            ResBlock2 += [Bottleneck(model_cx, model_x),]
            
        #======================== depth 1 ==========================
        ResBlock3 = []
        for k in range(23):
            model_x = []
            model_cx = []
            if k == 0:
                model_x = [model_ft.layer3[k].downsample[0], 
                           model_ft.layer3[k].downsample[1]]
            model_cx = [model_ft.layer3[k].conv1, 
                     model_ft.layer3[k].bn1,
                     model_ft.layer3[k].conv2,
                     model_ft.layer3[k].bn2,
                     model_ft.layer3[k].conv3,
                     model_ft.layer3[k].bn3]
            ResBlock3 += [Bottleneck(model_cx, model_x),]
        
        #======================== depth 0 ==========================
        ResBlock4 = []
        for m in range(3):
            model_x = []
            model_cx = []
            if m == 0:
                model_x = [model_ft.layer4[m].downsample[0], 
                           model_ft.layer4[m].downsample[1]]
                model_x[0].stride = (1, 1)
                           
            model_cx = [model_ft.layer4[m].conv1, 
                     model_ft.layer4[m].bn1,
                     model_ft.layer4[m].conv2,
                     model_ft.layer4[m].bn2,
                     model_ft.layer4[m].conv3,
                     model_ft.layer4[m].bn3]
            model_cx[2].stride = (1, 1)
            model_cx[2].dilation = (2, 2)
            model_cx[2].padding = (2, 2)
            ResBlock4 += [Bottleneck(model_cx, model_x),]
            
        #======================== depth -1 ==========================
        ResBlock5 = []
        conv_list = nn.ModuleList()
        conv1 = nn.Conv2d(inner_nc, outer_nc, kernel_size=1)
        conv_list.append(conv1)
        
        for n in range(1, 4):
            conv3 = nn.Conv2d(inner_nc, outer_nc, kernel_size=3)
            conv3.stride = (1, 1)
            conv3.dilation = (2 * n, 2 * n)
            conv3.padding = (2 * n, 2 * n)
            conv_list.append(conv3)
        ResBlock5 += [ASPP_Module(outer_nc, conv_list)]
         #======================== end =============================
         
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        
        uprelu = nn.ReLU(False)
        upnorm = norm_layer(outer_nc)
        
        if depth == 4:
            down = ResBlock0
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size = 4, stride = 2,
                                        padding = 1)
            up = [uprelu, upconv]
            model = down + [submodule] + up
            self.U4 = nn.Sequential(*model)
            
        if depth == 3:
            down = ResBlock1
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up
            self.U3 = nn.Sequential(*model)
            self.con3 = nn.Conv2d(outer_nc, outer_nc, kernel_size=1)
            self.con3.weight.data.normal_(0, 0.01)
            
        if depth == 2:
            down = ResBlock2
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up
            self.U2 = nn.Sequential(*model)
            self.con2 = nn.Conv2d(outer_nc, outer_nc, kernel_size=1)
            self.con2.weight.data.normal_(0, 0.01)
        
        if depth == 1:
            down = ResBlock3
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            up = [uprelu, upconv, upnorm]
            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up
            self.U1 = nn.Sequential(*model)
            self.con1 = nn.Conv2d(outer_nc, outer_nc, kernel_size=1)
            self.con1.weight.data.normal_(0, 0.01)
            
        if depth == 0:
            down = ResBlock4
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=3, stride=1,
                                        padding=1, bias=use_bias)
            up = [uprelu, upconv, upnorm]
            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up
            self.U0 = nn.Sequential(*model)
            self.con0 = nn.Conv2d(outer_nc, outer_nc, kernel_size=1)
            self.con0.weight.data.normal_(0, 0.01)
        
        if depth == -1: #idiot layer, forwards x straightly to next Unet block
            model = [submodule]
            self.U_1 = nn.Sequential(*model)
        
        if depth == -2: #model(x) forwards to Hook 
            down = ResBlock5
            #up = [nn.Upsample(256, mode='bilinear'),]
            lsm = [nn.LogSoftmax(),]
            model = down + lsm
            self.U_2 = nn.Sequential(*model)
                
    def forward(self, x):
        if self.depth == 4:
            sm = nn.Softmax2d()
            lsm = nn.LogSoftmax()
            t = self.U4(x)
            return {'GAN':sm(t * 5.0), 'L1':lsm(t)}
        elif self.depth == 3:
            return torch.cat([self.con3(x), self.U3(x)], 1)
        elif self.depth == 2:
            return torch.cat([self.con2(x), self.U2(x)], 1)
        elif self.depth == 1:
            return torch.cat([self.con1(x), self.U1(x)], 1)
        elif self.depth == 0:
            return torch.cat([self.con0(x), self.U0(x)], 1)
        elif self.depth == -1:
            _ = self.U_1(x)
            return x
        elif self.depth == -2:
            sm = nn.Softmax2d()
            lsm = nn.LogSoftmax()
            t = self.U_2(x)
            return {'GAN':sm(t * 5.0), 'L1':lsm(t)}
            #return self.U_2(x)


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[]):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            #use_bias = norm_layer.func == nn.InstanceNorm2d
            use_bias = False
        else:
            #use_bias = norm_layer == nn.InstanceNorm2d
            use_bias = False

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)
