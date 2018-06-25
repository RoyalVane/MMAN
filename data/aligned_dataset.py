import os.path
import random
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import math
import numpy as np
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image, ImageOps

def channel_1toN(img, num_channel):
    transform1 = transforms.Compose([transforms.ToTensor(),])
    img = (transform1(img) * 255.0).long()
    T = torch.LongTensor(num_channel, img.size(1), img.size(2)).zero_()
    #N = (torch.rand(num_channel, img.size(1), img.size(2)) - 0.5)/random.uniform(1e10, 1e25)#Noise
    mask = torch.LongTensor(img.size(1), img.size(2)).zero_()
    for i in range(num_channel):
        T[i] = T[i] + i
        layer = T[i] - img
        T[i] = torch.from_numpy(np.logical_not(np.logical_xor(layer.numpy(), mask.numpy())).astype(int))
    
# =============================================================================
#     T = T.float()+N
#     
#     S = T.sum(0)
#     for i in range(num_channel):
#         T[i] = torch.div(T[i],S)
# =============================================================================
    
    return T.float()

def channel_1to1(img):
    transform1 = transforms.Compose([transforms.ToTensor(),])
    T = torch.LongTensor(img.height, img.width).zero_()
    img = (transform1(img) * 255.0).long()
    T.resize_(img[0].size()).copy_(img[0])
    return T.long()
    
def swap_1(T, m, n): #Distinguish left & right
    A = T.numpy()
    m_mask = np.where(A == m, 1, 0)
    n_mask = np.where(A == n, 1, 0)
    A = A + (n - m)*m_mask + (m - n)*n_mask
    return torch.from_numpy(A)

def swap_N(T, m, n): #Distinguish left & right
    A = T.numpy()
    A[[m, n], :, :] = A[[n, m], :, :]
    return torch.from_numpy(A)
    
def get_label(T, num_channel):
    A = T.numpy()
    R = torch.FloatTensor(num_channel).zero_()
    for i in range(num_channel):
        if (A == i).any():
            R[i] = 1
    R = R[1:]
    return R

class parts_crop():
    def __init__(self, img, attribute):
        self.img = img
        self.attribute = attribute
        self.parts_bag = []
    
    def get_parts(self):
        array = np.asarray(self.img)
        for i in range(1, self.attribute.size(0)):
            w1 = 0
            w2 = array.shape[1] - 1
            h1 = 0
            h2 = array.shape[0] - 1
            if self.attribute[i]:
                while w1 < array.shape[1]:
                    if((array[:,w1] == i).any()):
                        break
                    w1  = w1 + 1
                
                while w2 > 0:
                    if((array[:,w2] == i).any()):
                        break
                    w2  = w2 - 1
                        
                while h1 < array.shape[0]:
                    if((array[h1,:] == i).any()):
                        break
                    h1  = h1 + 1
                        
                while h2 > 0:
                    if((array[h2,:] == i).any()):
                        break
                    h2  = h2 - 1
                    
                self.parts_bag.append(self.img.crop((w1, h1, w2, h2)))

class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase+ '_' + opt.dataset + '_A')
        self.A_paths = sorted(make_dataset(self.dir_A))
        self.dir_B = os.path.join(opt.dataroot, opt.phase+ '_' + opt.dataset + '_B')
        self.B_paths = sorted(make_dataset(self.dir_B))

        assert(len(self.A_paths) == len(self.B_paths))
        assert(opt.resize_or_crop == 'resize_and_crop')
        
        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.485, 0.456, 0.406),
                                               (0.229, 0.224, 0.225))]
        self.transform = transforms.Compose(transform_list)
        
    def __getitem__(self, index):

        A_path = self.A_paths[index]
        A = Image.open(A_path)
        A = A.resize((self.opt.loadSize , self.opt.loadSize), Image.LANCZOS)
        A_S = A.resize((int(self.opt.loadSize * 0.75), int(self.opt.loadSize * 0.75)), Image.LANCZOS)
        A_L = A.resize((int(self.opt.loadSize * 1.25), int(self.opt.loadSize * 1.25)), Image.LANCZOS)
        A_attribute = A.resize((int(self.opt.fineSize/16) , int(self.opt.fineSize/16)), Image.LANCZOS)
        
        #A = A.resize((int(self.opt.loadSize * (A.width/A.height)/16)*16, self.opt.loadSize), Image.LANCZOS) if A.width > A.height \
        #else A.resize((self.opt.loadSize , int(self.opt.loadSize  * (A.height/A.width)/16)*16), Image.LANCZOS)
        
        B_path = self.B_paths[index]
        B = Image.open(B_path)
        B = B.resize((self.opt.loadSize , self.opt.loadSize), Image.NEAREST)
        #B = B.resize((int(self.opt.loadSize * (B.width/B.height)/16)*16, self.opt.loadSize), Image.NEAREST) if B.width > B.height \
        #else B.resize((self.opt.loadSize , int(self.opt.loadSize  * (B.height/B.width)/16)*16), Image.NEAREST)
        
        if self.opt.loadSize > self.opt.fineSize:
            if random.random() < 0.4:
                area = A.size[0] * A.size[1]
                target_area = random.uniform(0.64, 1) * area
                aspect_ratio = random.uniform(4. / 5, 5. / 4)
    
                w = min(int(round(math.sqrt(target_area * aspect_ratio))), self.opt.loadSize)
                h = min(int(round(math.sqrt(target_area / aspect_ratio))), self.opt.loadSize)
    
                if random.random() < 0.5:
                    w, h = h, w
    
                if w <= A.size[0] and h <= A.size[1]:
                    x1 = random.randint(0, A.size[0] - w)
                    y1 = random.randint(0, A.size[1] - h)
    
                    A = A.crop((x1, y1, x1 + w, y1 + h))
                    B = B.crop((x1, y1, x1 + w, y1 + h))
                    assert(A.size == (w, h))
                
                A = A.resize((self.opt.fineSize , self.opt.fineSize), Image.LANCZOS)
                B = B.resize((self.opt.fineSize , self.opt.fineSize), Image.NEAREST)
            
            elif  0.4 < random.random() < 0.95:
                w_offset = random.randint(0, max(0, A.size[1] - self.opt.fineSize - 1))
                h_offset = random.randint(0, max(0, A.size[0] - self.opt.fineSize - 1))
                A = A.crop((w_offset, h_offset, w_offset + self.opt.fineSize, h_offset + self.opt.fineSize))
                B = B.crop((w_offset, h_offset, w_offset + self.opt.fineSize, h_offset + self.opt.fineSize))
            
            else:
                A = A.resize((self.opt.fineSize , self.opt.fineSize), Image.LANCZOS)
                B = B.resize((self.opt.fineSize , self.opt.fineSize), Image.NEAREST)
        
        A = self.transform(A)
        A_S = self.transform(A_S)
        A_L = self.transform(A_L)
        A_attribute = self.transform(A_attribute)

        B_L1 = channel_1to1(B)# single channel long tensor
        B_attribute_L1 = B.resize((int(self.opt.fineSize/16) , int(self.opt.fineSize/16)), Image.NEAREST)
        B = channel_1toN(B, self.opt.output_nc) # multi channel float tensor
        B_attribute_GAN = channel_1toN(B_attribute_L1, self.opt.output_nc) # multi channel float tensor for thumbnail
        B_attribute_L1 = channel_1to1(B_attribute_L1)
                
        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            idx_2 = [i for i in range(B_attribute_L1.size(1) - 1, -1, -1)]
            idx_2 = torch.LongTensor(idx_2)
            A = A.index_select(2, idx)
            A_attribute = A_attribute.index_select(2, idx_2)
            B_attribute_GAN = B_attribute_GAN.index_select(2, idx_2)
            B = B.index_select(2, idx)
            B_attribute_L1 = B_attribute_L1.index_select(1, idx_2)
            B_L1 = B_L1.index_select(1, idx)
            if self.opt.dataset == 'LIP':
                B = swap_N(B, 14, 15)
                B = swap_N(B, 16, 17)
                B = swap_N(B, 18, 19)
                B_attribute_GAN = swap_N(B_attribute_GAN, 14, 15)
                B_attribute_GAN = swap_N(B_attribute_GAN, 16, 17)
                B_attribute_GAN = swap_N(B_attribute_GAN, 18, 19)
                B_attribute_L1 = swap_1(B_attribute_L1, 14, 15)
                B_attribute_L1 = swap_1(B_attribute_L1, 16, 17)
                B_attribute_L1 = swap_1(B_attribute_L1, 18, 19)
                B_L1 = swap_1(B_L1, 14, 15)
                B_L1 = swap_1(B_L1, 16, 17)
                B_L1 = swap_1(B_L1, 18, 19)
                
        
        #construct missing part image
# =============================================================================
#         B_label = get_label(B_single, self.opt.output_nc)        
#         A_pose = torch.FloatTensor(B.size(0) - 1, B.size(1), B.size(2))
#         A_pose.copy_(B[1:, :, :])
#         for attempt in range (0, 10000):
#             u = random.randint(0, self.opt.output_nc - 2)
#             if B_label[u] == 1:
#                 A_pose[u, :, :].copy_(torch.FloatTensor(A_pose[u, :, :].size()).fill_(0))
#                 A_pose = torch.cat((A_pose, torch.zeros(A_pose.size(0), A_pose.size(1), A_pose.size(2))))
#                 A_pose[self.opt.output_nc - 1 + u, :, :].copy_(torch.FloatTensor(A_pose[self.opt.output_nc - 1 + u, :, :].size()).fill_(1))
#                 B_label.copy_(torch.FloatTensor(B_label.size()).fill_(0))
#                 B_label[u] = 1
#                 break
#             if attempt == 9999:
#                 A_pose = torch.cat((A_pose, torch.zeros(A_pose.size(0), A_pose.size(1), A_pose.size(2))))
#         B_pose = torch.FloatTensor(1, B.size(1), B.size(2))
#         B_pose.copy_(B[1+u, :, :])
#
#            
#        return {'A': A, 'A_S': A_S, 'A_L': A_L, 'A_pose': A_pose, 'B_pose' : B_pose, 'B_GAN': B, 'B_L1': B_single, 'B_Attribute': B_attribute, 'B_label': B_label, 
#               'A_paths': A_path, 'B_paths': B_path}
# =============================================================================
        return {'A': A, 'A_S': A_S, 'A_L': A_L, 'B_L1': B_L1, 'B_GAN': B, 
                'A_Attribute': A_attribute, 
                'B_Attribute_L1': B_attribute_L1, 
                'B_Attribute_GAN': B_attribute_GAN, 
                'A_paths': A_path, 'B_paths': B_path}
    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'AlignedDataset'
