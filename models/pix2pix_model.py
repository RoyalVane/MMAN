import numpy as np
import torch
import os
import random
import torch.nn as nn
from torchvision import models
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from . import pose_generator

def swap(T, m, n): #Distinguish left & right
    A = T.data.cpu().numpy()
    A[:, [m, n], :, :] = A[:, [n, m], :, :]
    return Variable(torch.from_numpy(A)).cuda()

class Pix2PixModel(BaseModel):
    def name(self):
        return 'Pix2PixModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.epoch = 1
        self.isTrain = opt.isTrain
        # define tensors for G1
        self.input_A = self.Tensor(opt.batchSize, opt.input_nc,
                                   opt.fineSize, opt.fineSize)
        self.input_A_S = self.Tensor(opt.batchSize, opt.input_nc,
                                   int(opt.fineSize * 0.75), int(opt.fineSize * 0.75))
        self.input_A_L = self.Tensor(opt.batchSize, opt.input_nc,
                                   int(opt.fineSize * 1.25), int(opt.fineSize * 1.25))
        
        self.input_A_Attribute = self.Tensor(opt.batchSize, opt.input_nc, int(opt.fineSize/16), int(opt.fineSize/16))
        
        self.input_B_GAN = self.Tensor(opt.batchSize, opt.output_nc,
                                   opt.fineSize, opt.fineSize)
        self.input_B_L1 = self.LongTensor(opt.batchSize,
                                   opt.fineSize, opt.fineSize)
        
        self.input_B_Attribute_GAN = self.Tensor(opt.batchSize, opt.output_nc, 
                                    int(opt.fineSize/16), int(opt.fineSize/16))
        
        self.input_B_Attribute_L1 = self.LongTensor(opt.batchSize, 
                                    int(opt.fineSize/16), int(opt.fineSize/16))
        
#        self.input_pose = self.Tensor(opt.batchSize, 19,
#                                    int(opt.fineSize/16), int(opt.fineSize/16))
        
        #G2
# =============================================================================
#         self.input_A_pose = self.Tensor(opt.batchSize, (opt.output_nc - 1) * 2, opt.fineSize, opt.fineSize)
#         
#         self.input_B_label = self.Tensor(opt.batchSize, opt.output_nc - 1, 1, 1)
#         
#         self.input_B_pose = self.Tensor(opt.batchSize, 1, opt.fineSize, opt.fineSize)
# =============================================================================
        
        #define hook
        self.hook = networks.UnetHook()
        
        # load/define networks
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                          opt.which_model_netG, self.hook, opt.fineSize, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        
        
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
# =============================================================================
#             self.netG2 = networks.define_G((opt.output_nc - 1) * 2, 1, opt.ngf,
#                                               'pose', None, opt.fineSize, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
# 
# =============================================================================
            #self.netG2 = pose_generator.construct_pose_model('./pretrained_models/coco_pose_iter_440000.pth.tar')
            #self.netG2 = pose_generator.construct_pose_model('/home/yawei/PedestrianParsing/Realtime_Multi-Person_Pose_Estimation-master/training/openpose_coco_best.pth.tar')
            
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf,
                                              opt.which_model_netD,
                                              opt.n_layers_D - 1, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
    
            self.netD2 = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf,
                                              opt.which_model_netD,
                                              opt.n_layers_D - 1, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
            
        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)
            if self.isTrain:
                #self.load_network(self.netG2, 'G2', opt.which_epoch)
                self.load_network(self.netD, 'D', opt.which_epoch)
                self.load_network(self.netD2, 'D2', opt.which_epoch)
                    

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr
            # define loss functions
            self.criterionL1 = torch.nn.NLLLoss2d()
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionAttributeL1 = torch.nn.NLLLoss2d()
            self.criterionAttributeGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)

            # initialize optimizers
            self.schedulers = []
            self.optimizers = []
            
            ignored_params = list(map(id, self.netG.model.U4[2].U3[4].U2[4].U1[23].U0[3].U_1[0].U_2.parameters() )) \
            + list(map(id, self.netG.model.U4[2].U3[4].U2[4].U1[23].U0[5].parameters() )) + list(map(id, self.netG.model.U4[2].U3[4].U2[4].U1[23].U0[6].parameters() )) + list(map(id, self.netG.model.U4[2].U3[4].U2[4].U1[23].con0.parameters() )) \
            + list(map(id, self.netG.model.U4[2].U3[4].U2[4].U1[25].parameters() )) + list(map(id, self.netG.model.U4[2].U3[4].U2[4].U1[26].parameters() )) + list(map(id, self.netG.model.U4[2].U3[4].U2[4].con1.parameters() )) \
            + list(map(id, self.netG.model.U4[2].U3[4].U2[6].parameters() )) + list(map(id, self.netG.model.U4[2].U3[4].U2[7].parameters() )) + list(map(id, self.netG.model.U4[2].U3[4].con2.parameters() )) \
            + list(map(id, self.netG.model.U4[2].U3[6].parameters() )) + list(map(id, self.netG.model.U4[2].U3[7].parameters() )) + list(map(id, self.netG.model.U4[2].con3.parameters() )) \
            + list(map(id, self.netG.model.U4[4].parameters() ))
            base_params = filter(lambda p: id(p) not in ignored_params, self.netG.parameters())
            self.optimizer_G = torch.optim.Adam([{'params': base_params, 'lr': 0.1 * opt.lr},
                                                {'params': self.netG.model.U4[2].U3[4].U2[4].U1[23].U0[3].U_1[0].U_2.parameters(), 'lr': opt.lr},
                                                {'params': self.netG.model.U4[2].U3[4].U2[4].U1[23].U0[5].parameters(), 'lr': opt.lr},
                                                {'params': self.netG.model.U4[2].U3[4].U2[4].U1[23].U0[6].parameters(), 'lr': opt.lr},
                                                {'params': self.netG.model.U4[2].U3[4].U2[4].U1[23].con0.parameters(), 'lr': opt.lr},
                                                {'params': self.netG.model.U4[2].U3[4].U2[4].U1[25].parameters(), 'lr': opt.lr},
                                                {'params': self.netG.model.U4[2].U3[4].U2[4].U1[26].parameters(), 'lr': opt.lr},
                                                {'params': self.netG.model.U4[2].U3[4].U2[4].con1.parameters(), 'lr': opt.lr},
                                                {'params': self.netG.model.U4[2].U3[4].U2[6].parameters(), 'lr': opt.lr},
                                                {'params': self.netG.model.U4[2].U3[4].U2[7].parameters(), 'lr': opt.lr},
                                                {'params': self.netG.model.U4[2].U3[4].con2.parameters(), 'lr': opt.lr},
                                                {'params': self.netG.model.U4[2].U3[6].parameters(), 'lr': opt.lr},
                                                {'params': self.netG.model.U4[2].U3[7].parameters(), 'lr': opt.lr},
                                                {'params': self.netG.model.U4[2].con3.parameters(), 'lr': opt.lr},
                                                {'params': self.netG.model.U4[4].parameters(), 'lr': opt.lr},
                                                ], betas=(opt.beta1, 0.999), weight_decay = 1e-4)
            
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay = 1e-4)
            
            self.optimizer_D2 = torch.optim.Adam(self.netD2.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay = 1e-4)
            
            
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_D2)
            
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        if self.isTrain:
            networks.print_network(self.netD)
            networks.print_network(self.netD2)
        print('-----------------------------------------------')

# =============================================================================
#     input_A: original color image
#     input_B_GAN: original label image (multiple channels) for GAN loss calculation
#     input_B_L1: original label image (single channel) for L1 loss calculation
#     input_B_Attribute: original thumbnail for Attribute loss calculation
# =============================================================================
    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        
        #G1
        input_A = input['A']
        input_A_S = input['A_S']
        input_A_L = input['A_L']
        input_A_Attribute = input['A_Attribute']
        
        input_B_GAN = input['B_GAN']
        input_B_L1 = input['B_L1']
        input_B_Attribute_L1 = input['B_Attribute_L1']
        input_B_Attribute_GAN = input['B_Attribute_GAN']
        #pose_dic = pose_generator.process(self.netG2, input_A, 16)
        #input_pose = pose_dic['pose_map']
        
        
        #G2
#        input_A_pose = input['A_pose']
#        input_B_label = input['B_label']
#        input_B_pose = input['B_pose']
        
        #G1
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_A_S.resize_(input_A_S.size()).copy_(input_A_S)
        self.input_A_L.resize_(input_A_L.size()).copy_(input_A_L)
        self.input_A_Attribute.resize_(input_A_Attribute.size()).copy_(input_A_Attribute)
        
        self.input_B_GAN.resize_(input_B_GAN.size()).copy_(input_B_GAN)
        self.input_B_L1.resize_(input_B_L1.size()).copy_(input_B_L1)
        self.input_B_Attribute_GAN.resize_(input_B_Attribute_GAN.size()).copy_(input_B_Attribute_GAN)
        self.input_B_Attribute_L1.resize_(input_B_Attribute_L1.size()).copy_(input_B_Attribute_L1)
        #self.input_pose.resize_(input_pose.size()).copy_(input_pose)
        #self.pose_num = pose_dic['total_points']
        
        #G2
#        self.input_A_pose.resize_(input_A_pose.size()).copy_(input_A_pose)
#        self.input_B_label.resize_(input_B_label.size()).copy_(input_B_label)
#        self.input_B_pose.resize_(input_B_pose.size()).copy_(input_B_pose)
        
    def forward(self):
        self.real_A = Variable(self.input_A)
        self.real_A_Attribute = Variable(self.input_A_Attribute)

        #Copy from files
        self.real_B_GAN = Variable(self.input_B_GAN) #multi-channel target for label map
        self.real_B_L1 = Variable(self.input_B_L1) #single-channel target for label map
        self.real_B_Attribute_GAN = Variable(self.input_B_Attribute_GAN) #multi-channel target for thumbnail
        self.real_B_Attribute_L1 = Variable(self.input_B_Attribute_L1) # single-channel target for thumbnail
        #self.real_B_pose = Variable(self.input_B_pose)
        
        #Generate from networks
        self.fake_B_GAN = self.netG(self.real_A)['GAN'] #multi-channel label map--> target real_B_GAN
        self.fake_B_L1 = self.netG(self.real_A)['L1'] #multi-channel label map but nagtive --> target real_B_L1
        self.fake_B_Attribute_GAN = self.hook.get_value()['GAN'] #multi-channel thumbnail --> real_B_Attribute_GAN
        self.fake_B_Attribute_L1 = self.hook.get_value()['L1'] #multi-channel thumbnail but nagtive --> real_B_Attribute_L1
        
        #self.fake_B_pose = self.netG2(self.real_A_pose)
        
        #Generate from Both network
# =============================================================================
#         u = self.input_B_label.max(1)[1]
#         real_slice = torch.FloatTensor(1, 1, self.opt.fineSize, self.opt.fineSize)
#         pose_slice = torch.FloatTensor(1, 1, self.opt.fineSize, self.opt.fineSize)
#         fake_slice = torch.FloatTensor(1, 1, self.opt.fineSize, self.opt.fineSize)
#         change_slice = torch.FloatTensor(1, 1, self.opt.fineSize, self.opt.fineSize)
#         pose_target = torch.FloatTensor(1, self.opt.output_nc, self.opt.fineSize, self.opt.fineSize)
#         real_slice.copy_(self.real_B_GAN.data[:, u[0] + 1, :, :])
#         pose_slice.copy_(self.fake_B_pose.data.ge(0.1))
#         fake_slice.copy_(self.fake_B_GAN.data[:, u[0] + 1, :, :])
#         pose_target.copy_(self.fake_B_GAN.data)
#         and_slice = torch.clamp(torch.add(real_slice, 1,  pose_slice), max = 1.0)
#         torch.addcmul(change_slice, 1, and_slice, fake_slice)
#         pose_target[:, u[0] + 1, :, :].copy_(change_slice)
#         self.pose_target = Variable(pose_target).cuda()
# =============================================================================
        
        
    # no backprop gradients
    def test(self):
        self.real_A = Variable(self.input_A, volatile=True)
        self.real_A_S = Variable(self.input_A_S, volatile=True)
        self.real_A_L = Variable(self.input_A_L, volatile=True)

        M = nn.Upsample(int(self.real_A.size(3)), mode='bilinear')

        self.fake_B_GAN = self.netG(self.real_A)['GAN']
        self.fake_B_GAN_S = self.netG(self.real_A_S)['GAN']
        self.fake_B_GAN_L = self.netG(self.real_A_L)['GAN']

        self.fake_B_GAN_S = M(self.fake_B_GAN_S)
        self.fake_B_GAN_L = M(self.fake_B_GAN_L)
        
        idx = [i for i in range(self.real_A.size(3) - 1, -1, -1)]
        idx = torch.LongTensor(idx).cuda()
        idx = Variable(idx)
        idx_S = [i for i in range(self.real_A_S.size(3) - 1, -1, -1)]
        idx_S = torch.LongTensor(idx_S).cuda()
        idx_S = Variable(idx_S)
        idx_L = [i for i in range(self.real_A_L.size(3) - 1, -1, -1)]
        idx_L = torch.LongTensor(idx_L).cuda()
        idx_L = Variable(idx_L)
        
        self.real_A_flip = self.real_A.index_select(3, idx)
        self.fake_B_flip = self.netG(self.real_A_flip)['GAN']
        self.fake_B_flip_flip = self.fake_B_flip.index_select(3, idx)
        if self.opt.dataset == 'LIP':
            self.fake_B_flip_flip = swap(self.fake_B_flip_flip, 14, 15)
            self.fake_B_flip_flip = swap(self.fake_B_flip_flip, 16, 17)
            self.fake_B_flip_flip = swap(self.fake_B_flip_flip, 18, 19)
        
        self.real_A_flip_S = self.real_A_S.index_select(3, idx_S)
        self.fake_B_flip_S = self.netG(self.real_A_flip_S)['GAN']
        self.fake_B_flip_flip_S = self.fake_B_flip_S.index_select(3, idx_S)
        if self.opt.dataset == 'LIP':
            self.fake_B_flip_flip_S = swap(self.fake_B_flip_flip_S, 14, 15)
            self.fake_B_flip_flip_S = swap(self.fake_B_flip_flip_S, 16, 17)
            self.fake_B_flip_flip_S = swap(self.fake_B_flip_flip_S, 18, 19)
        self.fake_B_flip_flip_S = M(self.fake_B_flip_flip_S)
        
        self.real_A_flip_L = self.real_A_L.index_select(3, idx_L)
        self.fake_B_flip_L = self.netG(self.real_A_flip_L)['GAN']
        self.fake_B_flip_flip_L = self.fake_B_flip_L.index_select(3, idx_L)
        if self.opt.dataset == 'LIP':
            self.fake_B_flip_flip_L = swap(self.fake_B_flip_flip_L, 14, 15)
            self.fake_B_flip_flip_L = swap(self.fake_B_flip_flip_L, 16, 17)
            self.fake_B_flip_flip_L = swap(self.fake_B_flip_flip_L, 18, 19)
        self.fake_B_flip_flip_L = M(self.fake_B_flip_flip_L)
        
        self.fake_B_GAN = self.fake_B_GAN + self.fake_B_flip_flip + self.fake_B_GAN_S + self.fake_B_flip_flip_S + self.fake_B_GAN_L + self.fake_B_flip_flip_L
        self.real_B_GAN = Variable(self.input_B_GAN, volatile=True)

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B_GAN), 1).data)
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        
        # Real
        real_AB = torch.cat((self.real_A, self.real_B_GAN), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        
        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    
    def backward_D2(self):
        fake_AB_Attribute = torch.cat((self.real_A_Attribute, self.fake_B_Attribute_GAN), 1)
        pred_fake_Attribute = self.netD2(fake_AB_Attribute.detach())
        self.loss_D_fake_Attribute = self.criterionAttributeGAN(pred_fake_Attribute, False)
        
        real_AB_Attribute = torch.cat((self.real_A_Attribute, self.real_B_Attribute_GAN), 1)
        pred_real_Attribute = self.netD2(real_AB_Attribute)
        self.loss_D_real_Attribute = self.criterionAttributeGAN(pred_real_Attribute, True)
        
        self.loss_D_Attribute = (self.loss_D_fake_Attribute + self.loss_D_real_Attribute) * 0.5
        self.loss_D_Attribute.backward()
        
        
    def backward_G(self):
        #GAN loss: G(x) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B_GAN), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        
        #Attribute GAN loss: G(x) should fake the discriminator2
        fake_AB_Attribute = torch.cat((self.real_A_Attribute, self.fake_B_Attribute_GAN), 1)
        pred_fake_Attribute = self.netD2(fake_AB_Attribute.detach())
        self.loss_G_GAN_Attribute = self.criterionAttributeGAN(pred_fake_Attribute, True)
        
        #L1 loss: Minimize logSoftmax concats NLL2d between original size
        self.loss_G_L1 = self.criterionL1(self.fake_B_L1, self.real_B_L1) * self.opt.lambda_A
        
        #Attribute L1 loss: Minimize logSoftmax concats NLL2d between thumbnail
        self.loss_G_L1_Attribute = self.criterionAttributeL1(self.fake_B_Attribute_L1, self.real_B_Attribute_L1) * self.opt.lambda_A
        
        #Total loss
# =============================================================================
#         self.loss_G = min(self.epoch/20.0, 5.0) * self.loss_G_L1 + 1.0 * self.loss_G_L1_Attribute +  \
#             min(self.epoch//50, 0.0) * self.loss_G_GAN + min(self.epoch//50, 1.0) * self.loss_G_GAN_Attribute #for LIP
# =============================================================================
        
        self.loss_G = 5.0 * self.loss_G_L1 + 1.0 * self.loss_G_L1_Attribute +  \
           1 * self.loss_G_GAN + 1 * self.loss_G_GAN_Attribute #for LIP
            
# =============================================================================
#         self.loss_G = 5 * self.loss_G_L1 + 1.0 * self.loss_G_L1_Attribute
# =============================================================================
            
# =============================================================================
#         self.loss_G = min(self.epoch/2.0, 5.0) * self.loss_G_L1 + 1.0 * self.loss_G_L1_Attribute +  \
#             1.0 * self.loss_G_GAN + 1.0 * self.loss_G_GAN_Attribute #for Pascal
# =============================================================================
            
        self.loss_G.backward()
        
# =============================================================================
#     def backward_G2(self):
#         fake_AB_pose = torch.cat((self.real_A_pose, self.fake_B_pose), 1)
#         pred_fake_pose = self.netD2(fake_AB_pose)
#         self.loss_G_pose_GAN = self.criterionGAN2(pred_fake_pose, True)
#         self.loss_G_pose_L1 = self.criterionL1_pose(self.fake_B_pose, self.real_B_pose)
#         self.loss_G_pose = self.loss_G_pose_GAN + 0.75 * self.loss_G_pose_L1
#         self.loss_G_pose.backward()
# =============================================================================
        
    def optimize_parameters(self):
        self.forward()

        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D2.zero_grad()
        self.backward_D2()
        
        if self.epoch > -1: #for Pascal
        #if self.epoch > 48: #for LIP
            if random.random() < 0.1:
                self.optimizer_D.step()
                self.optimizer_D2.step()
            
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        
    

    def get_current_errors(self):

        return OrderedDict([('G_GAN', self.loss_G_GAN.data[0]),
                            ('G_L1', self.loss_G_L1.data[0]),
                            ('D_real', self.loss_D_real.data[0]),
                            ('D_fake', self.loss_D_fake.data[0]),
                            ('G_GAN_Attri', self.loss_G_GAN_Attribute.data[0]),
                            ('G_L1_Attri', self.loss_G_L1_Attribute.data[0]),
                            ('D_real_Attri', self.loss_D_real_Attribute.data[0]),
                            ('D_fake_Attri', self.loss_D_fake_Attribute.data[0])
                            ])

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.ndim_tensor2im(self.fake_B_GAN.data, dataset = self.opt.dataset)
        real_B = util.ndim_tensor2im(self.real_B_GAN.data, dataset = self.opt.dataset)
        #pose = util.ndim_tensor2im(self.real_pose.data, dataset = self.opt.dataset)
        #real_A_pose = util.ndim_tensor2im(self.real_A_pose.data[:, 0:self.opt.output_nc - 1, :, :], dataset = self.opt.dataset, dim = 'pose')
        #fake_B_pose = util.onedim_tensor2im(self.fake_B_pose.data, self.input_B_label.max(1)[1], dataset = self.opt.dataset)
        #real_B_pose = util.onedim_tensor2im(self.real_B_pose.data, self.input_B_label.max(1)[1], dataset = self.opt.dataset)
        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B)])

    def save(self, label):
        self.epoch = self.epoch + 1
        print('weight L1:', min(self.epoch / 20.0, 5.0) * self.opt.lambda_A, 'weight L1 Attri:', self.opt.lambda_A, 
              'weight GAN:', min(self.epoch // 20.0, 1.0), 'weight GAN Attri:', 0.5)
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netD, 'D', label, self.gpu_ids)
        self.save_network(self.netD2, 'D2', label, self.gpu_ids)
