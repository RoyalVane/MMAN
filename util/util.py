from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import inspect, re
import os
import collections

# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array

def tensor2im(image_tensor, imtype=np.uint8):
    #print(image_tensor)
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_numpy = (std * image_numpy + mean) * 255
    return image_numpy.astype(imtype)

def ndim_tensor2im(image_tensor, imtype=np.uint8, dataset = 'PPSS', dim = 'L2'):
    if dataset == 'Horse' or dataset == 'Cow' or dataset == 'PPSS':
        #palette_idx = np.array([[0, 0, 0], [0, 32, 255], [0, 191, 255], [96, 255, 159], [255, 80, 0], [255, 255, 0], [175, 0, 0], [143, 0, 0]]) #PPSS
         palette_idx = np.array([[0, 0, 143], [0, 32, 255], [0, 32, 255], [255, 80, 0], [0, 191, 255], [96, 255, 159], [96, 255, 159], [96, 255, 159], [143, 0, 0], [255, 255, 0],
              [96, 255, 159], [96, 255, 159], [255, 255, 0], [0, 191, 255], [255, 80, 0], [255, 80, 0], [175, 0, 0], [175, 0, 0], [143, 0, 0], [143, 0, 0]])#LIP
    if dataset == 'LIP' or dataset == 'Market':
        palette_idx = np.array([[0, 0, 0], [0, 255, 255], [255, 0, 255], [255, 255, 0], [255, 170, 255], [255, 255, 170], [170, 255, 255], [85, 255, 255], [85, 170, 255], [105, 45, 190],
              [170, 105, 255], [170, 255, 85], [255, 170, 85], [255, 85, 170],  [85, 255, 85], [0, 255, 85], [255, 0, 85], [255, 85, 85], [0, 85, 255], [85, 85, 255]])#LIP
    if dataset == 'Pascal':
        palette_idx = np.array([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128], [0, 128, 128], [85, 255, 255], [85, 170, 255], [105, 45, 190],
              [170, 105, 255], [170, 255, 85], [255, 170, 85], [255, 85, 170],  [85, 255, 85], [0, 255, 85], [255, 0, 85], [255, 85, 85], [0, 85, 255], [85, 85, 255]])#Pascal
    if dataset == 'Market2':
        palette_idx = np.array([[0, 0, 0], [255, 0, 255], [255, 0, 255], [0, 255, 85], [255, 85, 170], [255, 255, 170], [255, 255, 170], [255, 255, 170], [255, 85, 85], [255, 170, 85],
              [255, 255, 170], [255, 255, 170], [255, 170, 85], [255, 85, 170],  [0, 255, 85], [0, 255, 85], [255, 85, 85], [255, 85, 85], [85, 85, 255], [85, 85, 255]])#LIP
    result = np.zeros(shape = (image_tensor.size(2), image_tensor.size(3), 3))
    image_numpy = image_tensor[0].cpu().float().numpy()
    #image_numpy = image_tensor[0].data.numpy()
    for i in range(image_numpy.shape[1]):
        for j in range(image_numpy.shape[2]):
            if dim == 'L2':
                result[i][j] = palette_idx[np.argmax(image_numpy[:,i,j])]
            elif dim == 'pose':
                result[i][j] = palette_idx[np.argmax(image_numpy[:,i,j]) + 1]
    return result.astype(imtype)
    
def ndim_tensor2im2(image_tensor, imtype=np.uint8, dataset = 'PPSS', dim = 'L2'):
    if dataset == 'Horse' or dataset == 'Cow' or dataset == 'PPSS':
        palette_idx = np.array([[0, 0, 0], [0, 32, 255], [0, 191, 255], [96, 255, 159], [255, 80, 0], [255, 255, 0], [175, 0, 0], [143, 0, 0]]) #PPSS
# =============================================================================
#          palette_idx = np.array([[0, 0, 143], [0, 32, 255], [0, 32, 255], [255, 80, 0], [0, 191, 255], [96, 255, 159], [96, 255, 159], [96, 255, 159], [143, 0, 0], [255, 255, 0],
#               [96, 255, 159], [96, 255, 159], [255, 255, 0], [0, 191, 255], [255, 80, 0], [255, 80, 0], [175, 0, 0], [175, 0, 0], [143, 0, 0], [143, 0, 0]])#LIP
# =============================================================================
    if dataset == 'LIP':
        palette_idx = np.array([[0, 0, 0], [0, 255, 255], [255, 0, 255], [255, 255, 0], [255, 170, 255], [255, 255, 170], [170, 255, 255], [85, 255, 255], [85, 170, 255], [105, 45, 190],
              [170, 105, 255], [170, 255, 85], [255, 170, 85], [255, 85, 170],  [85, 255, 85], [0, 255, 85], [255, 0, 85], [255, 85, 85], [0, 85, 255], [85, 85, 255]])#LIP
    if dataset == 'Pascal':
        palette_idx = np.array([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128], [0, 128, 128], [85, 255, 255], [85, 170, 255], [105, 45, 190],
              [170, 105, 255], [170, 255, 85], [255, 170, 85], [255, 85, 170],  [85, 255, 85], [0, 255, 85], [255, 0, 85], [255, 85, 85], [0, 85, 255], [85, 85, 255]])#Pascal
    
    result = np.zeros(shape = (image_tensor.size(2), image_tensor.size(3), 3))
    image_numpy = image_tensor[0].cpu().float().numpy()
    #image_numpy = image_tensor[0].data.numpy()
    for i in range(image_numpy.shape[1]):
        for j in range(image_numpy.shape[2]):
            if dim == 'L2':
                result[i][j] = palette_idx[np.argmax(image_numpy[:,i,j])]
            elif dim == 'pose':
                result[i][j] = palette_idx[np.argmax(image_numpy[:,i,j]) + 1]
    return result.astype(imtype)

def onedim_tensor2im(image_tensor, imtype=np.uint8, dataset = 'PPSS'):
    if dataset == 'PPSS':
        palette_idx = np.array([[0, 0, 143], [0, 32, 255], [0, 191, 255], [96, 255, 159], [255, 80, 0], [255, 255, 0], [175, 0, 0], [143, 0, 0]]) #PPSS
    if dataset == 'LIP':
        palette_idx = np.array([[0, 0, 0], [0, 255, 255], [255, 0, 255], [255, 255, 0], [255, 170, 255], [255, 255, 170], [170, 255, 255], [85, 255, 255], [85, 170, 255], [105, 45, 190],
              [170, 105, 255], [170, 255, 85], [255, 170, 85], [255, 85, 170],  [85, 255, 85], [0, 255, 85], [255, 0, 85], [255, 85, 85], [0, 85, 255], [85, 85, 255]])#LIP
    if dataset == 'Pascal':
        palette_idx = np.array([[0, 0, 0], [255, 0, 0], [155, 100, 0], [128, 128, 0], [0, 128, 128], [0, 100, 155], [0, 0, 255]])#Pascal
    result = np.zeros(shape = (image_tensor.size(1), image_tensor.size(2), 3))
    #image_numpy = image_tensor[0].cpu().float().numpy()
    for i in range(image_tensor.size(1)):
        for j in range(image_tensor.size(2)):
            #result[i][j] = palette_idx[np.argmax(image_numpy[:,i,j]) + 1]
            if image_tensor.data[0][i][j] > 0.8:
                result[i][j] = palette_idx[1]
            elif 0.65 < image_tensor.data[0][i][j] < 0.8:
                result[i][j] = palette_idx[2]
            elif 0.5 < image_tensor.data[0][i][j] < 0.65:
                result[i][j] = palette_idx[3]
            elif 0.35 < image_tensor.data[0][i][j] < 0.5:
                result[i][j] = palette_idx[4]
            elif 0.2 < image_tensor.data[0][i][j] < 0.35:
                result[i][j] = palette_idx[5]
            else:
                result[i][j] = palette_idx[6]
    return result.astype(imtype)

def pose_tensor2im(image_tensor, imtype=np.uint8, dataset = 'Pascal'):
    
    result = np.zeros(shape = (image_tensor.size(2), image_tensor.size(3), 3))
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = ([0.5,] * image_numpy + [0.5,]) * 255
    for i in range(image_numpy.shape[1]):
        for j in range(image_numpy.shape[2]):
            result[i][j] = [image_numpy[0][i][j], image_numpy[0][i][j], image_numpy[0][i][j]]
    
    return result.astype(imtype)

def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def info(object, spacing=10, collapse=1):
    """Print methods and doc strings.
    Takes module, class, list, dictionary, or string."""
    methodList = [e for e in dir(object) if isinstance(getattr(object, e), collections.Callable)]
    processFunc = collapse and (lambda s: " ".join(s.split())) or (lambda s: s)
    print( "\n".join(["%s %s" %
                     (method.ljust(spacing),
                      processFunc(str(getattr(object, method).__doc__)))
                     for method in methodList]) )

def varname(p):
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
        if m:
            return m.group(1)

def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
