import os

import cv2
import numpy as np
from PIL import Image
import PIL
from skimage.io import imsave
from natsort import natsorted
from tqdm import  tqdm

def img_save(image,imagename,savepath):
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    # Gray_pic
    imsave(os.path.join(savepath, "{}.png".format(imagename)),image)


def img2RGB(f_name, vi_name):
    vi_img = Image.open(vi_name)
    vi_img = vi_img.convert('YCbCr')
    f_img = Image.open(f_name).convert('L')
    vi_Y, vi_Cb, vi_Cr = vi_img.split()
    f_img = Image.open(f_name).convert('L')
    f_img = np.dstack((f_img, vi_Cr, vi_Cb))
    # f_img = Image.merge('YCbCr', (f_img, vi_Cb, vi_Cr))
    # f_RGB = f_img.convert('RGB')
    f_RGB = cv2.cvtColor(f_img, cv2.COLOR_YCrCb2RGB)
    img_save(f_RGB, f_name.split('/')[-1].split('.')[0], './RGB')

if __name__ == '__main__':
    fusion_folder = './test_result/M3FD' ## 融合图像所在的文件夹
    vi_filoder = './test_img/M3FD/vi' ## 可见光图像所在的文件夹 两个文件夹里的图片命名需要保持一致
    file_list = os.listdir(fusion_folder)
    file_bar = tqdm(file_list)
    for file in natsorted(file_bar):
        f_name = os.path.join(fusion_folder, file)
        vi_name = os.path.join(vi_filoder, file)
        img2RGB(f_name, vi_name)
        file_bar.set_description('Y2RGB %s' % file)