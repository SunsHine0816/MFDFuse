import torch.utils.data as Data
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader


class H5Dataset(Data.Dataset):
    def __init__(self, h5file_path):
        self.h5file_path = h5file_path
        h5f = h5py.File(h5file_path, 'r')
        self.keys = list(h5f['ir_patchs'].keys()) #字典获得keys？
        h5f.close()

    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, index):
        h5f = h5py.File(self.h5file_path, 'r')
        key = self.keys[index]
        IR = np.array(h5f['ir_patchs'][key])
        VIS = np.array(h5f['vis_patchs'][key])
        h5f.close()
        return torch.Tensor(VIS), torch.Tensor(IR)

# trainloader = DataLoader(H5Dataset(r'../data/MSRS_train_imgsize_128_stride_200.h5'),
#                          batch_size=8,
#                          shuffle=True,
#                          num_workers=0)
#batch_size为8表示每次取八张图片为一组
# print(trainloader.__len__()) 60×8
# for i, (data_VIS, data_IR) in enumerate(trainloader):
#     print(data_IR.size())
