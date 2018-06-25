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

small_scale = 0.75
large_scale = 1.25

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
    return T.float()


def channel_1to1(img):
    transform1 = transforms.Compose([transforms.ToTensor(),])
    T = torch.LongTensor(img.height, img.width).zero_()
    img = (transform1(img) * 255.0).long()
    T.resize_(img[0].size()).copy_(img[0])
    return T.long()


def Image_dropout(img, thres):
    if random.random() > thres:
        return img
    else:
        img = np.zeros((img.height, img.width))
        img = Image.fromarray(np.uint8(img))
        return img
    
    
def Image_dropout2(img, thres):
    random_vec = torch.LongTensor(19)
    for i in range(19):
        if random.random() > thres:
            random_vec[i] = 1
        else:
            random_vec[i] = 0
    random_vec[0] = 0

    A = np.asarray(img)
    mask = np.zeros((img.height, img.width))
    for k in range(19):
        if random_vec[k] > 0.5:
            mask.fill(k * 14)
            mask = np.uint8(np.logical_not(np.logical_not(A - mask)))
            A = A * mask
    return Image.fromarray(np.uint8(A))
    
    
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


class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase+ '_' + opt.dataset + '_A')
        self.A_paths = sorted(make_dataset(self.dir_A))
        self.dir_B = os.path.join(opt.dataroot, opt.phase+ '_' + opt.dataset + '_B')
        self.B_paths = sorted(make_dataset(self.dir_B))
        self.dir_P = os.path.join(opt.dataroot, opt.phase+ '_' + opt.dataset + '_POSE')
        self.P_paths = sorted(make_dataset(self.dir_P))
        
        if self.opt.dataset == 'Pascal':
            a = np.loadtxt('Pascal_scale.txt')
            self.Scale_list = torch.from_numpy(a)
        
        assert(len(self.A_paths) == len(self.B_paths))
        assert(opt.resize_or_crop == 'resize_and_crop')
        
        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.485, 0.456, 0.406),
                                               (0.229, 0.224, 0.225))]
                          
        transform_list2 = [transforms.ToTensor(),
                          transforms.Normalize((0.5,),
                                               (0.5,))]
                          
        self.transform = transforms.Compose(transform_list)
        self.transform2 = transforms.Compose(transform_list2)
        
    def __getitem__(self, index):

        Scale = self.Scale_list[index]
        if Scale == 0:
            Scale = 10
        Scale_small = min(int(40 / Scale + 2) * 0.25, 2.0)
        Scale_large = min(int(40 / Scale + 4) * 0.25, 2.0)
        
        A_path = self.A_paths[index]
        A = Image.open(A_path)
        A = A.resize((self.opt.loadSize , self.opt.loadSize), Image.LANCZOS)
        A_S = A.resize((int(self.opt.loadSize * Scale_small), int(self.opt.loadSize * Scale_small)), Image.LANCZOS)
        A_L = A.resize((int(self.opt.loadSize * Scale_large), int(self.opt.loadSize * Scale_large)), Image.LANCZOS)
        
        #A = A.resize((int(self.opt.loadSize * (A.width/A.height)/16)*16, self.opt.loadSize), Image.LANCZOS) if A.width > A.height \
        #else A.resize((self.opt.loadSize , int(self.opt.loadSize  * (A.height/A.width)/16)*16), Image.LANCZOS)
        
        B_path = self.B_paths[index]
        B = Image.open(B_path)
        B = B.resize((self.opt.loadSize , self.opt.loadSize), Image.NEAREST)
        
        P_path = self.P_paths[index]
        P = Image.open(P_path)
        P = P.resize((self.opt.loadSize , self.opt.loadSize), Image.NEAREST)
        P_S = P.resize((int(self.opt.loadSize * Scale_small), int(self.opt.loadSize * Scale_small)), Image.NEAREST)
        P_L = P.resize((int(self.opt.loadSize * Scale_large), int(self.opt.loadSize * Scale_large)), Image.NEAREST)
            
        
        #B = B.resize((int(self.opt.loadSize * (B.width/B.height)/16)*16, self.opt.loadSize), Image.NEAREST) if B.width > B.height \
        #else B.resize((self.opt.loadSize , int(self.opt.loadSize  * (B.height/B.width)/16)*16), Image.NEAREST)
        
        if self.opt.loadSize > self.opt.fineSize:
            P = Image_dropout2(P, 1)
            if random.random() < 0.99:
                area = A.size[0] * A.size[1]
                target_area = random.uniform(0.36, 1) * area
                aspect_ratio = random.uniform(3. / 4, 4. / 3)
    
                w = min(int(round(math.sqrt(target_area * aspect_ratio))), self.opt.loadSize)
                h = min(int(round(math.sqrt(target_area / aspect_ratio))), self.opt.loadSize)
    
                if random.random() < 0.5:
                    w, h = h, w
    
                if w <= A.size[0] and h <= A.size[1]:
                    x1 = random.randint(0, A.size[0] - w)
                    y1 = random.randint(0, A.size[1] - h)
    
                    A = A.crop((x1, y1, x1 + w, y1 + h))
                    B = B.crop((x1, y1, x1 + w, y1 + h))
                    P = P.crop((x1, y1, x1 + w, y1 + h))
                    assert(A.size == (w, h))
                
                    A = A.resize((self.opt.fineSize , self.opt.fineSize), Image.LANCZOS)
                    B = B.resize((self.opt.fineSize , self.opt.fineSize), Image.NEAREST)
                    P = P.resize((self.opt.fineSize , self.opt.fineSize), Image.NEAREST)
                
                else:
                    w_offset = random.randint(0, max(0, A.size[1] - self.opt.fineSize - 1))
                    h_offset = random.randint(0, max(0, A.size[0] - self.opt.fineSize - 1))
                    A = A.crop((w_offset, h_offset, w_offset + self.opt.fineSize, h_offset + self.opt.fineSize))
                    B = B.crop((w_offset, h_offset, w_offset + self.opt.fineSize, h_offset + self.opt.fineSize))
                    P = P.crop((w_offset, h_offset, w_offset + self.opt.fineSize, h_offset + self.opt.fineSize))
                    
            else:
                A = A.resize((self.opt.fineSize , self.opt.fineSize), Image.LANCZOS)
                B = B.resize((self.opt.fineSize , self.opt.fineSize), Image.NEAREST)
                P = P.resize((self.opt.fineSize , self.opt.fineSize), Image.NEAREST)
        
        A_attribute = A.resize((int(self.opt.fineSize/16) , int(self.opt.fineSize/16)), Image.LANCZOS)
        A = self.transform(A)
        A_S = self.transform(A_S)
        A_L = self.transform(A_L)
        A_attribute = self.transform(A_attribute)

        B_L1 = channel_1to1(B)# single channel long tensor
        B_attribute_L1 = B.resize((int(self.opt.fineSize/16) , int(self.opt.fineSize/16)), Image.NEAREST)
        B = channel_1toN(B, self.opt.output_nc) # multi channel float tensor
        B_attribute_GAN = channel_1toN(B_attribute_L1, self.opt.output_nc) # multi channel float tensor for thumbnail
        B_attribute_L1 = channel_1to1(B_attribute_L1)
                
        P_attribute = P.resize((int(self.opt.fineSize/16) , int(self.opt.fineSize/16)), Image.NEAREST)
        P = self.transform2(P)
        P_S = self.transform2(P_S)
        P_L = self.transform2(P_L)
        P_attribute = self.transform2(P_attribute)
        
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
            B_attribute_GAN = B_attribute_GAN.index_select(2, idx_2)
            B = B.index_select(2, idx)
            B_attribute_L1 = B_attribute_L1.index_select(1, idx_2)
            B_L1 = B_L1.index_select(1, idx)
            P = P.index_select(2, idx)
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
# =============================================================================
#                 P = swap_N(P, 3, 6)
#                 P = swap_N(P, 4, 7)
#                 P = swap_N(P, 5, 8)
#                 P = swap_N(P, 9, 12)
#                 P = swap_N(P, 10, 13)
#                 P = swap_N(P, 11, 14)
#                 P = swap_N(P, 15, 16)
#                 P = swap_N(P, 17, 18)
#                 
#                 P_attribute = swap_N(P_attribute, 3, 6)
#                 P_attribute = swap_N(P_attribute, 4, 7)
#                 P_attribute = swap_N(P_attribute, 5, 8)
#                 P_attribute = swap_N(P_attribute, 9, 12)
#                 P_attribute = swap_N(P_attribute, 10, 13)
#                 P_attribute = swap_N(P_attribute, 11, 14)
#                 P_attribute = swap_N(P_attribute, 15, 16)
#                 P_attribute = swap_N(P_attribute, 17, 18)
# =============================================================================

        
        return {'A': A, 'A_S': A_S, 'A_L': A_L, 'A_Attribute': A_attribute, 'B_L1': B_L1, 'B_GAN': B, 
                'P': P, 'P_Attribute': P_attribute, 'P_S': P_S, 'P_L': P_L,
                'B_Attribute_L1': B_attribute_L1, 'B_Attribute_GAN': B_attribute_GAN, 
                'A_paths': A_path, 'B_paths': B_path, 'Scale_small': Scale_small, 'Scale_large': Scale_large}
    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'AlignedDataset'
