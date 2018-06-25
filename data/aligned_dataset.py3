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

class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase+ '_' + opt.dataset + '_A')
        self.A_paths = sorted(make_dataset(self.dir_A))

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
                    assert(A.size == (w, h))
                
                A = A.resize((self.opt.fineSize , self.opt.fineSize), Image.LANCZOS)
            
            elif  0.4 < random.random() < 0.95:
                w_offset = random.randint(0, max(0, A.size[1] - self.opt.fineSize - 1))
                h_offset = random.randint(0, max(0, A.size[0] - self.opt.fineSize - 1))
                A = A.crop((w_offset, h_offset, w_offset + self.opt.fineSize, h_offset + self.opt.fineSize))
            
            else:
                A = A.resize((self.opt.fineSize , self.opt.fineSize), Image.LANCZOS)
        
        A = self.transform(A)
        A_S = self.transform(A_S)
        A_L = self.transform(A_L)
        A_attribute = self.transform(A_attribute)

        
        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

                
        return {'A': A, 'A_S': A_S, 'A_L': A_L, 
                'A_Attribute': A_attribute, 
                'A_paths': A_path}
    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'AlignedDataset'
