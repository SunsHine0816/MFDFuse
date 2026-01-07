# -*- coding: utf-8 -*-

'''
---------

---------------------------------------------------------------------
Import packages
------------------------------------------------------------------------------
'''
import numpy as np

from net import Restormer_Encoder, Restormer_Decoder, BaseFeatureExtraction, DetailFeatureExtraction, \
    feature_measurement, feature_weight_map, feature_enhancement
from utils.dataset import H5Dataset
import os

# 设置环境变量，防止多次加载库，既然你执迷不悟，那就签免责声明吧
os.environ['KMP_D   UPLICATE_LIB_OK'] = 'True'
# system
import sys
import time
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.loss import Fusionloss, cc
import kornia

'''
------------------------------------------------------------------------------
Configure our network
------------------------------------------------------------------------------
'''
# 指定GPU`
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
criteria_fusion = Fusionloss()
model_str = 'MFDFuse'
# . Set the hyper-parameters for training
num_epochs = 120  # total epoch
epoch_gap = 40 # epoches of Phase I

lr = 1e-4
weight_decay = 0
batch_size = 4
GPU_number = os.environ['CUDA_VISIBLE_DEVICES']
# Coefficients of the loss func tion
coeff_mse_loss_VF = 1.
coeff_mse_loss_IF = 1.
coeff_decomp = 5.
coeff_tv = 5.
# coeff_tv = 6
coeff_fusionloss = 4.

clip_grad_norm_value = 0.01
optim_step = 20
optim_gamma = 0.5

# Model
device = 'cuda' if torch.cuda.is_available() else 'cpu'

DIDF_Encoder = nn.DataParallel(Restormer_Encoder()).to(device)
DIDF_Decoder = nn.DataParallel(Restormer_Decoder()).to(device)
BaseFuseLayer = nn.DataParallel(BaseFeatureExtraction(dim=64, num_heads=8)).to(device)
BaseFuseLayer_s = nn.DataParallel(BaseFeatureExtraction(dim=64, num_heads=8)).to(device)
DetailFuseLayer = nn.DataParallel(DetailFeatureExtraction(num_layers=1)).to(device)
FeatureMeasurement = nn.DataParallel(feature_measurement()).to(device)
FeatureWeightMap = nn.DataParallel(feature_weight_map()).to(device)
FeatureEnhancement = nn.DataParallel(feature_enhancement()).to(device)

# optimizer, scheduler and loss function

optimizer1 = torch.optim.Adam(
    DIDF_Encoder.parameters(), lr=lr, weight_decay=weight_decay)
optimizer2 = torch.optim.Adam(
    DIDF_Decoder.parameters(), lr=lr, weight_decay=weight_decay)
optimizer3 = torch.optim.Adam(
    BaseFuseLayer.parameters(), lr=lr, weight_decay=weight_decay)
optimizer7 = torch.optim.Adam(
    BaseFuseLayer_s.parameters(), lr=lr, weight_decay=weight_decay)
optimizer4 = torch.optim.Adam(
    DetailFuseLayer.parameters(), lr=lr, weight_decay=weight_decay)
optimizer5 = torch.optim.Adam(
    FeatureWeightMap.parameters(), lr=lr, weight_decay=weight_decay)
optimizer6 = torch.optim.Adam(
    FeatureEnhancement.parameters(), lr=lr, weight_decay=weight_decay)

# 根据epoch训练次数来调整学习率
# step_size每经过多少个数据批次做一次学习率更新，gamma更新lr的乘法因子
scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=optim_step, gamma=optim_gamma)
scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=optim_step, gamma=optim_gamma)
scheduler3 = torch.optim.lr_scheduler.StepLR(optimizer3, step_size=optim_step, gamma=optim_gamma)
scheduler4 = torch.optim.lr_scheduler.StepLR(optimizer4, step_size=optim_step, gamma=optim_gamma)
scheduler5 = torch.optim.lr_scheduler.StepLR(optimizer5, step_size=optim_step, gamma=optim_gamma)
scheduler6 = torch.optim.lr_scheduler.StepLR(optimizer6, step_size=optim_step, gamma=optim_gamma)
scheduler7 = torch.optim.lr_scheduler.StepLR(optimizer7, step_size=optim_step, gamma=optim_gamma)
# 损失函数
MSELoss = nn.MSELoss()
L1Loss = nn.L1Loss()
Loss_ssim = kornia.losses.SSIM(window_size=11, reduction='mean')

# data loader
trainloader = DataLoader(H5Dataset(r"data/MSRS_train_imgsize_128_stride_200.h5"),
                         batch_size=batch_size,
                         shuffle=True,
                         num_workers=0)

loader = {'train': trainloader, }
timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")

'''
------------------------------------------------------------------------------
Train
------------------------------------------------------------------------------
'''

step = 0

'''
设置 torch.backends.cudnn.benchmark=True 将会让程序在开始时花费一点额外时间，为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速。
设置这个 flag 可以让内置的 cuDNN 的 auto-tuner 自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题
'''
torch.backends.cudnn.benchmark = True
prev_time = time.time()
w1_V_all = list([])
w2_I_all = list([])
w1 = 0
w2 = 0

for epoch in range(num_epochs):
    ''' train '''
    for i, (data_VIS, data_IR) in enumerate(loader['train']):
        # 指定使用GPU
        data_VIS, data_IR = data_VIS.cuda(), data_IR.cuda()  # [8, 1, 128, 128]
        # 设置为训练模式
        DIDF_Encoder.train()
        DIDF_Decoder.train()
        BaseFuseLayer.train()
        DetailFuseLayer.train()
        FeatureWeightMap.train()
        FeatureEnhancement.train()
        BaseFuseLayer_s.train()
        # 清空梯度
        DIDF_Encoder.zero_grad()
        DIDF_Decoder.zero_grad()
        BaseFuseLayer.zero_grad()
        DetailFuseLayer.zero_grad()
        FeatureWeightMap.zero_grad()
        FeatureEnhancement.zero_grad()
        BaseFuseLayer_s.zero_grad()
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()
        optimizer4.zero_grad()
        optimizer5.zero_grad()
        optimizer6.zero_grad()
        optimizer7.zero_grad()

        if epoch < epoch_gap:  # Phase I
            feature_V_B, feature_V_D, _,  = DIDF_Encoder(data_VIS)
            feature_I_B, feature_I_D, _,  = DIDF_Encoder(data_IR)
            weight_list = FeatureMeasurement(torch.cat((feature_V_B, feature_V_D), dim=1),
                                             torch.cat((feature_I_B, feature_I_D), dim=1))
            data_VIS_hat, _ = DIDF_Decoder(data_VIS, feature_V_B, feature_V_D, None, None)
            data_IR_hat, _ = DIDF_Decoder(data_IR, feature_I_B, feature_I_D, None, None)
            # 自定义损失函数
            cc_loss_B = cc(feature_V_B, feature_I_B)
            cc_loss_D = cc(feature_V_D, feature_I_D)
            # L vis
            mse_loss_V = weight_list[:, 0][0] * (
                        5 * Loss_ssim(data_VIS, data_VIS_hat) + 3 * MSELoss(data_VIS, data_VIS_hat))
            # L ir
            mse_loss_I = weight_list[:, 1][0] * (5 * Loss_ssim(data_IR, data_IR_hat) + 3 * MSELoss(data_IR, data_IR_hat))

            w1_V_all.append(weight_list[:, 0][0])
            w2_I_all.append(weight_list[:, 1][0])

            # 梯度损失
            Gradient_loss = L1Loss(kornia.filters.SpatialGradient()(data_VIS),
                                   kornia.filters.SpatialGradient()(data_VIS_hat))
            # L decomp
            loss_decomp = (cc_loss_D) ** 2 / (1.01 + cc_loss_B)
            # L total
            # loss = coeff_mse_loss_VF * mse_loss_V + coeff_mse_loss_IF * \
            #        mse_loss_I + coeff_decomp * loss_decomp + coeff_tv * Gradient_loss
            loss = mse_loss_V + mse_loss_I + coeff_decomp * loss_decomp + coeff_tv * Gradient_loss
            # 看Phase II
            loss.backward()
            nn.utils.clip_grad_norm_(
                DIDF_Encoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                DIDF_Decoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            optimizer1.step()
            optimizer2.step()
        else:  # Phase II
            feature_V_B, feature_V_D, feature_V = DIDF_Encoder(data_VIS)
            feature_I_B, feature_I_D, feature_I= DIDF_Encoder(data_IR)
            feature_F_B = BaseFuseLayer(feature_I_B + feature_V_B)
            feature_F_D = DetailFuseLayer(feature_I_D + feature_V_D)
            low_frequency_F, high_frequency_F = FeatureEnhancement(data_VIS, data_IR)
            feature_vis, feature_ir = FeatureWeightMap(data_VIS, data_IR)

            # 传入了可将光图像
            data_Fuse, feature_F = DIDF_Decoder(data_VIS, feature_F_B + low_frequency_F, feature_F_D + high_frequency_F,
                                                feature_vis, feature_ir)

            w1 = torch.mean(torch.tensor(w1_V_all)).item()
            w2 = torch.mean(torch.tensor(w2_I_all)).item()

            cc_loss_B = cc(feature_V_B, feature_I_B)
            cc_loss_D = cc(feature_V_D, feature_I_D)
            loss_decomp = (cc_loss_D) ** 2 / (1.01 + cc_loss_B)
            fusionloss, loss_in, lossgrad, lossLgrad = criteria_fusion(data_VIS, data_IR, data_Fuse)
            MSE_SSIM_loss_V = w1 / (w1 + w2) * (5 * Loss_ssim(data_VIS, data_Fuse) + 3 * MSELoss(data_VIS, data_Fuse))
            MSE_SSIM_loss_I = w2 / (w1 + w2) * (5 * Loss_ssim(data_IR, data_Fuse) + 3 * MSELoss(data_IR, data_Fuse))

            loss =  fusionloss + coeff_decomp * loss_decomp + MSE_SSIM_loss_I + MSE_SSIM_loss_V + lossLgrad
            # 计算梯度→裁剪梯度→更新网络参数
            # 计算梯度
            loss.backward()
            # 对所有的梯度乘以一个clip_coef，clip_coef = max_norm / total_norm
            # total_norm受norm_type的影响
            # 裁剪梯度
            nn.utils.clip_grad_norm_(
                DIDF_Encoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                DIDF_Decoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                BaseFuseLayer.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                DetailFuseLayer.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                FeatureWeightMap.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                FeatureEnhancement.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                BaseFuseLayer_s.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            # 更新网络参数
            optimizer1.step()
            optimizer2.step()
            optimizer3.step()
            optimizer4.step()
            optimizer5.step()
            optimizer6.step()
            optimizer7.step()
        # Determine approximate time left
        batches_done = epoch * len(loader['train']) + i
        batches_left = num_epochs * len(loader['train']) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [loss: %f] ETA: %.10s"
            % (
                epoch,
                num_epochs,
                i,
                len(loader['train']),
                loss.item(),
                time_left,
            )
        )

    # adjust the learning rate
    # Phase I
    scheduler1.step()
    scheduler2.step()
    # Phase II
    if not epoch < epoch_gap:
        scheduler3.step()
        scheduler4.step()
        scheduler5.step()
        scheduler6.step()
        scheduler7.step()
    # 避免学习率过低
    if optimizer1.param_groups[0]['lr'] <= 1e-6:
        optimizer1.param_groups[0]['lr'] = 1e-6
    if optimizer2.param_groups[0]['lr'] <= 1e-6:
        optimizer2.param_groups[0]['lr'] = 1e-6
    if optimizer3.param_groups[0]['lr'] <= 1e-6:
        optimizer3.param_groups[0]['lr'] = 1e-6
    if optimizer4.param_groups[0]['lr'] <= 1e-6:
        optimizer4.param_groups[0]['lr'] = 1e-6
    if optimizer5.param_groups[0]['lr'] <= 1e-6:
        optimizer5.param_groups[0]['lr'] = 1e-6
    if optimizer6.param_groups[0]['lr'] <= 1e-6:
        optimizer6.param_groups[0]['lr'] = 1e-6
    if optimizer7.param_groups[0]['lr'] <= 1e-6:
        optimizer7.param_groups[0]['lr'] = 1e-6

if True:
    # 保存模型
    checkpoint = {
        'DIDF_Encoder': DIDF_Encoder.state_dict(),
        'DIDF_Decoder': DIDF_Decoder.state_dict(),
        'BaseFuseLayer': BaseFuseLayer.state_dict(),
        'BaseFuseLayer_s': BaseFuseLayer_s.state_dict(),
        'DetailFuseLayer': DetailFuseLayer.state_dict(),
        'FeatureWeightMap': FeatureWeightMap.state_dict(),
        'FeatureEnhancement': FeatureEnhancement.state_dict(),
    }
    torch.save(checkpoint, os.path.join("models/MFDFuse_" + timestamp + '.pth'))


