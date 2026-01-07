import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia as k
from numpy.linalg import norm
import numpy as np

#计算Lgrad+Lint
from torch.autograd import Variable


class Fusionloss(nn.Module):
    def __init__(self):
        super(Fusionloss, self).__init__()
        self.sobelconv=Sobelxy()
        self.alpha3 = 10 # 10
        self.alpha4 = 13 # 13

        self.Laplace_gradient = Laplace_gradient()

    def forward(self, image_vis, image_ir, generate_img):
        image_y=image_vis[:,:1,:,:]
        x_in_max=torch.max(image_y,image_ir)
        loss_in=F.l1_loss(x_in_max,generate_img)
        y_grad=self.sobelconv(image_y) #vis gradient
        ir_grad=self.sobelconv(image_ir) #ir gradient
        generate_img_grad=self.sobelconv(generate_img) #fuse gradient
        x_grad_joint=torch.max(y_grad,ir_grad)
        loss_grad=F.l1_loss(x_grad_joint,generate_img_grad)
        loss_total=self.alpha4 * loss_in + self.alpha3 * loss_grad

        vis_len_grad = self.sobelconv(image_vis+image_vis-LoG(image_vis))
        ir_hen_grad = self.sobelconv(image_ir+LoG(image_ir))
        fuse_len_grad = self.Laplace_gradient(generate_img+generate_img-LoG(generate_img))
        fuse_hen_grad = self.Laplace_gradient(generate_img+LoG(generate_img))
        loss_Lgrad = 8 * torch.norm(fuse_len_grad-vis_len_grad, p=2) + 6 * torch.norm(fuse_hen_grad-ir_hen_grad, p=2)
        loss_Lgrad = 1/(128*128)*loss_Lgrad

        return loss_total,loss_in,loss_grad,loss_Lgrad

class Sobelxy(nn.Module):
    #从不同方向计算梯度
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
    def forward(self, x):
        #            input    卷积核
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)

class Laplace_gradient(nn.Module):
    def __init__(self):
        super(Laplace_gradient, self).__init__()
        kernelL = [[0,1,0],
                   [1,-4,1],
                   [0,1,0]]
        kernelL = torch.FloatTensor(kernelL).unsqueeze(0).unsqueeze(0)
        kernelL = 1/8 * kernelL
        self.weightx = nn.Parameter(data=kernelL, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernelL, requires_grad=False).cuda()
    def forward(self, x):
        sobelx = F.conv2d(x, self.weightx, padding=1)
        sobely = F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)

def minmaxscaler(data):
    min = torch.min(data)
    max = torch.max(data)
    return (data - min) / (max - min)

def LoG(img):
    window_size = 9
    window = torch.Tensor([[[0, 1, 1, 2, 2, 2, 1, 1, 0],
                                 [1, 2, 4, 5, 5, 5, 4, 2, 1],
                                 [1, 4, 5, 3, 0, 3, 5, 4, 1],
                                 [2, 5, 3, -12, -24, -12, 3, 5, 2],
                                 [2, 5, 0, -24, -40, -24, 0, 5, 2],
                                 [2, 5, 3, -12, -24, -12, 3, 5, 2],
                                 [1, 4, 5, 3, 0, 3, 4, 4, 1],
                                 [1, 2, 4, 5, 5, 5, 4, 2, 1],
                                 [0, 1, 1, 2, 2, 2, 1, 1, 0]]]).cuda()
    #img1_array = np.array(img, dtype=np.float32)  # Image -> array
    #img1_tensor = torch.from_numpy(img1_array)  # array -> tensor
    channel = img.shape[1]
    window = Variable(window.expand(channel, 1, window_size, window_size).contiguous())
    output = F.conv2d(img, window, padding=window_size // 2, groups=channel)
    output = minmaxscaler(output)  # 归一化到0~1之间
    return output # Log(X(x,y))

def cc(img1, img2):
    eps = torch.finfo(torch.float32).eps
    """Correlation coefficient for (N, C, H, W) image; torch.float32 [0.,1.]."""
    #batch_size,channel
    N, C, _, _ = img1.shape
    img1 = img1.reshape(N, C, -1)
    img2 = img2.reshape(N, C, -1)
    cosine = torch.sum(img1 * img2, dim=-1) / (eps + torch.sqrt(torch.sum(img1 **
                                                                      2, dim=-1)) * torch.sqrt(torch.sum(img2 ** 2, dim=-1)))
    #torch.clamp(input, min, max, out=None)
    cosine = torch.clamp(cosine, -1., 1.)
    return cosine.mean()

# class DensefeatureLoss(nn.Module):
#     def __init__(self):
#         super(DensefeatureLoss, self).__init__()
#         self.sobelconv = Sobelxy()
#         self.alpha1 = 1
#         self.alpha2 = 1
#     def forward(self, image_vis, image_ir, vis_feature, ir_feature):
#         loss_in = F.l1_loss(image_ir, ir_feature)
#         vis_grad = self.sobelconv(image_vis)
#         vfeature_grad = self.sobelconv(vis_feature)
#         loss_grad = torch.norm(vis_grad - vfeature_grad, p=2)
#         loss_total = self.alpha1 * loss_in + self.alpha2 * loss_grad
#         return loss_total
def y_to_xyz(y):
    """
    将Y值（假设来自YCrCb）转换为XYZ值
    参数:
    y (numpy.ndarray): 形状为 (height, width) 的Y值数组
    返回:
    xyz (numpy.ndarray): 形状为 (height, width, 3) 的XYZ值数组
    """
    # 这里假设Y值范围是0 - 255，将其转换为0 - 1范围
    # y = y / 255.0
    # 因为是灰度值，假设R = G = B = Y
    # [6,1,128,128]
    r = y
    g = y
    b = y
    x = r * 0.412453 + g * 0.357580 + b * 0.180423
    y = r * 0.212671 + g * 0.715160 + b * 0.072169
    z = r * 0.019334 + g * 0.119193 + b * 0.950227
    xyz = torch.cat((x, y, z),dim=1)

    return xyz


def xyz_to_l(xyz):
    """
    将XYZ值转换为CIELAB中的L值
    参数:
    xyz (numpy.ndarray): 形状为 (height, width, 3) 的XYZ值数组
    返回:
    l (numpy.ndarray): 形状为 (height, width) 的L值数组
    """
    x_n = 95.047
    y_n = 100.000
    z_n = 108.883

    y = xyz[:, :, 1] / y_n

    l = torch.where(y > 0.008856, 116 * (y ** (1 / 3)) - 16, 903.3 * y)
    return l


def y_to_l(y):
    """
    将Y值（来自YCrCb）转换为CIELAB中的L值
    参数:
    y (numpy.ndarray): 形状为 (height, width) 的Y值数组
    返回:
    l (numpy.ndarray): 形状为 (height, width) 的L值数组
    """
    xyz = y_to_xyz(y)
    l = xyz_to_l(xyz)
    return l

def lab_loss(img1, img2):

    lab_1 = y_to_l(img1)
    lab_2 = y_to_l(img2)

    return F.l1_loss(lab_1, lab_2)
