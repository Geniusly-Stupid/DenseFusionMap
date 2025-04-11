from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
import skimage
import skimage.io
import skimage.transform
import numpy as np
import time
from submodels import *
from dataloader import preprocess
from PIL import Image

# ==================================================
# Patch adaptative_cat so that it crops inputs to a common spatial size
def adaptative_cat(*tensors):
    min_h = min(t.size(2) for t in tensors)
    min_w = min(t.size(3) for t in tensors)
    cropped = [t[:, :, :min_h, :min_w] for t in tensors]
    return torch.cat(cropped, dim=1)

# Replace the adaptative_cat function in depthCompleNew with our version
import submodels.depthCompleNew as dc
dc.adaptative_cat = adaptative_cat

# ==================================================
# Monkey-patch tensor addition to handle mismatched spatial dimensions:
# Instead of only cropping the second operand, we crop both operands
# to the common minimal size in the spatial dimensions.
orig_tensor_add = torch.Tensor.__add__
def safe_tensor_add(self, other):
    if self.dim() == 4 and other.dim() == 4:
        h = min(self.size(2), other.size(2))
        w = min(self.size(3), other.size(3))
        self_cropped = self[:, :, :h, :w]
        other_cropped = other[:, :, :h, :w]
        return orig_tensor_add(self_cropped, other_cropped)
    return orig_tensor_add(self, other)
torch.Tensor.__add__ = safe_tensor_add

# Similarly patch torch.add
orig_torch_add = torch.add
def safe_torch_add(a, b):
    if a.dim() == 4 and b.dim() == 4:
        h = min(a.size(2), b.size(2))
        w = min(a.size(3), b.size(3))
        a_cropped = a[:, :, :h, :w]
        b_cropped = b[:, :, :h, :w]
        return orig_torch_add(a_cropped, b_cropped)
    return orig_torch_add(a, b)
torch.add = safe_torch_add
# ==================================================

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description='deepCpmpletion')
parser.add_argument('--loadmodel', default='depth_completion_KITTI.tar',
                    help='load model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

model = s2dN(1)

# If CUDA is enabled, wrap the model in DataParallel and move it to GPU
if args.cuda:
    model = nn.DataParallel(model, device_ids=[0])
    model.cuda()

modelpath = os.path.join(ROOT_DIR, args.loadmodel)

if args.loadmodel is not None:
    if args.cuda:
        state_dict = torch.load(modelpath)["state_dict"]
    else:
        state_dict = torch.load(modelpath, map_location=torch.device('cpu'))["state_dict"]
    
    # Remove "module." prefix from state dict keys if present
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("module.", "")
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict)
print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

def test(imgL, sparse, mask):
    model.eval()
    
    if args.cuda:
       imgL = torch.FloatTensor(imgL).cuda()
       sparse = torch.FloatTensor(sparse).cuda()
       mask = torch.FloatTensor(mask).cuda()
    else:
       imgL = torch.FloatTensor(imgL)
       sparse = torch.FloatTensor(sparse)
       mask = torch.FloatTensor(mask)

    imgL = Variable(imgL)
    sparse = Variable(sparse)
    mask = Variable(mask)

    start_time = time.time()
    with torch.no_grad():
        outC, outN, maskC, maskN = model(imgL, sparse, mask)

    tempMask = torch.zeros_like(outC)
    predC = outC[:, 0, :, :]
    predN = outN[:, 0, :, :]
    tempMask[:, 0, :, :] = maskC
    tempMask[:, 1, :, :] = maskN
    # Apply softmax along dimension 1
    predMask = F.softmax(tempMask, dim=1)
    predMaskC = predMask[:, 0, :, :]
    predMaskN = predMask[:, 1, :, :]
    pred1 = predC * predMaskC + predN * predMaskN
    time_temp = time.time() - start_time

    output1 = torch.squeeze(pred1)

    return output1.data.cpu().numpy(), time_temp
      
def rmse(gt, img, ratio):
    dif = gt[np.where(gt > ratio)] - img[np.where(gt > ratio)]
    error = np.sqrt(np.mean(dif**2))
    return error

def mae(gt, img, ratio):
    dif = gt[np.where(gt > ratio)] - img[np.where(gt > ratio)]
    error = np.mean(np.fabs(dif))
    return error

def irmse(gt, img, ratio):
    dif = 1.0/gt[np.where(gt > ratio)] - 1.0/img[np.where(gt > ratio)]
    error = np.sqrt(np.mean(dif**2))
    return error

def imae(gt, img, ratio):
    dif = 1.0/gt[np.where(gt > ratio)] - 1.0/img[np.where(gt > ratio)]
    error = np.mean(np.fabs(dif))
    return error

def main():
   processed = preprocess.get_transform(augment=False)

   # gt_fold = r'D:\Desktop\EECS568\Project\DenseFusionMap\data\depth\proj_depth\groundtruth\image_02'
   # left_fold = r'D:\Desktop\EECS568\Project\DenseFusionMap\data\depth\image_02\data'
   # lidar2_raw = r'D:\Desktop\EECS568\Project\DenseFusionMap\data\depth\velodyne_raw\image_02'
   
   # output_dir = r'D:\Desktop\EECS568\Project\DenseFusionMap\data\depth\depth_map'
   
   gt_fold = r'D:\Desktop\EECS568\Project\DenseFusionMap\data\slam\projected_depth'
   left_fold = r'D:\Desktop\EECS568\Project\DenseFusionMap\data\slam\rgb'
   lidar2_raw = r'D:\Desktop\EECS568\Project\DenseFusionMap\data\slam\projected_depth'
   
   output_dir = r'D:\Desktop\EECS568\Project\DenseFusionMap\data\slam\depth'

   gt = [img for img in os.listdir(gt_fold)]
   image = [img for img in os.listdir(left_fold)]
   lidar2 = [img for img in os.listdir(lidar2_raw)]
   gt_test = [os.path.join(gt_fold, img) for img in gt]
   left_test = [os.path.join(left_fold, img) for img in image]
   sparse2_test = [os.path.join(lidar2_raw, img) for img in lidar2]
   
   print("gt: ", gt_test[0])
   print("rgb: ", left_test[0])
   print("laser: ", sparse2_test[0])
   
   left_test.sort()
   sparse2_test.sort()
   gt_test.sort()

   time_all = 0.0

   for inx in range(len(left_test)):
       print("index: ", inx)

       imgL_o = skimage.io.imread(left_test[inx])
       if imgL_o.ndim == 2:
           # If grayscale, duplicate channels to form a 3-channel image
           imgL_o = np.stack((imgL_o,)*3, axis=-1)
       imgL = processed(imgL_o).numpy()
       imgL = np.reshape(imgL, [1, 3, imgL_o.shape[0], imgL_o.shape[1]])

       gtruth = skimage.io.imread(gt_test[inx]).astype(np.float32)
       gtruth = gtruth * 1.0 / 256.0
       sparse = skimage.io.imread(sparse2_test[inx]).astype(np.float32)
       sparse = sparse * 1.0 / 256.0

       mask = np.where(sparse > 0.0, 1.0, 0.0)
       mask = np.reshape(mask, [imgL_o.shape[0], imgL_o.shape[1], 1])
       sparse = np.reshape(sparse, [imgL_o.shape[0], imgL_o.shape[1], 1])
       sparse = processed(sparse).numpy()
       sparse = np.reshape(sparse, [1, 1, imgL_o.shape[0], imgL_o.shape[1]])
       mask = processed(mask).numpy()
       mask = np.reshape(mask, [1, 1, imgL_o.shape[0], imgL_o.shape[1]])
       
       output1 = os.path.join(output_dir, os.path.basename(left_test[inx]))

       pred, time_temp = test(imgL, sparse, mask)
       pred = np.where(pred <= 0.0, 0.9, pred)

       time_all = time_all + time_temp
       print("time: ", time_temp)

       pred_show = pred * 256.0
       pred_show = pred_show.astype('uint16')
       res_buffer = pred_show.tobytes()
       img_out = Image.new("I", pred_show.T.shape)
       img_out.frombytes(res_buffer, 'raw', "I;16")
       img_out.save(output1)
       print("Saved to: ", output1)

   print("time: %.8f" % (time_all * 1.0 / 1000.0))

if __name__ == '__main__':
   main()
   # Optionally, restore the original addition operators after testing:
   torch.Tensor.__add__ = orig_tensor_add
   torch.add = orig_torch_add
