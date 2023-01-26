# from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# from PIL import Image
# import matplotlib.pyplot as plt

# import torchvision.transforms as transforms
import torchvision.models as models

import copy
import numpy as np
import pandas as pd
import os
# import scipy.io as sio

# from torch.autograd import Variable
# from torchvision.transforms import ToTensor

from torch.utils.data import Dataset, DataLoader
# from torchvision import datasets
from pathlib import Path

class MyDataset(Dataset):
    def __init__(self, ydata_df, data_dir, device):#='./DNN_data/data/'
        # ydata_df = 
        # ydata_df = pd.read_csv(yfile_name, index_col=0)
        # x = xdata_df.values#.iloc[:,0:8]
        
        self.device = device
        
        self.gene = ydata_df.iloc[:,0].values
        target_y = ydata_df.iloc[:,1:].values
        
        # self.x_train = torch.tensor(x, dtype=torch.float32)
        self.y_train = torch.tensor(target_y, dtype=torch.float32)
        self.y_train = self.y_train.to(self.device)
        self.data_dir = data_dir
 
    def __len__(self):
        return(len(self.gene))
   
    def __getitem__(self, idx):
        slice_path = os.path.join(self.data_dir, self.gene[idx] + '.npy')
        slice_in = np.load(slice_path)
        
        self.x_train = torch.tensor(slice_in)
        self.x_train = self.x_train.to(self.device)
            
        return(self.x_train, self.y_train[idx])
    
## from pytorch
import sys
import re
import shutil
import tempfile
import hashlib
from urllib.request import urlopen, Request
from urllib.parse import urlparse
HASH_REGEX = re.compile(r'-([a-f0-9]*)\.')

class _Faketqdm(object):  # type: ignore[no-redef]

    def __init__(self, total=None, disable=False,
                 unit=None, *args, **kwargs):
        self.total = total
        self.disable = disable
        self.n = 0
        # Ignore all extra *args and **kwargs lest you want to reinvent tqdm

    def update(self, n):
        if self.disable:
            return

        self.n += n
        if self.total is None:
            sys.stderr.write("\r{0:.1f} bytes".format(self.n))
        else:
            sys.stderr.write("\r{0:.1f}%".format(100 * self.n / float(self.total)))
        sys.stderr.flush()

    def close(self):
        self.disable = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.disable:
            return

        sys.stderr.write('\n')

try:
    from tqdm import tqdm  # If tqdm is installed use it, otherwise use the fake wrapper
except ImportError:
    tqdm = _Faketqdm

def download_url_to_file(url, file_name, hash_prefix=None, progress=True):
    r"""Download object at the given URL to a local path.
    Args:
        url (str): URL of the object to download
        dst (str): Full path where object will be saved, e.g. ``/tmp/temporary_file``
        hash_prefix (str, optional): If not None, the SHA256 downloaded file should start with ``hash_prefix``.
            Default: None
        progress (bool, optional): whether or not to display a progress bar to stderr
            Default: True
    Example:
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_HUB)
        >>> # xdoctest: +REQUIRES(POSIX)
        >>> torch.hub.download_url_to_file('https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth', '/tmp/temporary_file')
    """
    file_size = None
    req = Request(url, headers={"User-Agent": "torch.hub"})
    u = urlopen(req)
    meta = u.info()
    if hasattr(meta, 'getheaders'):
        content_length = meta.getheaders("Content-Length")
    else:
        content_length = meta.get_all("Content-Length")
    if content_length is not None and len(content_length) > 0:
        file_size = int(content_length[0])

    # We deliberately save it in a temp file and move it after
    # download is complete. This prevents a local working checkpoint
    # being overridden by a broken download.
    
    dst_dir = os.path.dirname(__file__)
    dst = os.path.expanduser(os.path.join(dst_dir, file_name))
    f = tempfile.NamedTemporaryFile(delete=False, dir=dst_dir)

    try:
        if hash_prefix is not None:
            sha256 = hashlib.sha256()
        with tqdm(total=file_size, disable=not progress,
                  unit='B', unit_scale=True, unit_divisor=1024) as pbar:
            while True:
                buffer = u.read(8192)
                if len(buffer) == 0:
                    break
                f.write(buffer)
                if hash_prefix is not None:
                    sha256.update(buffer)
                pbar.update(len(buffer))

        f.close()
        if hash_prefix is not None:
            digest = sha256.hexdigest()
            if digest[:len(hash_prefix)] != hash_prefix:
                raise RuntimeError('invalid hash value (expected "{}", got "{}")'
                                   .format(hash_prefix, digest))
        shutil.move(f.name, dst)
    finally:
        f.close()
        if os.path.exists(f.name):
            os.remove(f.name)



class NbcNet(models.ResNet):#()nn.Module
    def __init__(self, para):
        super(NbcNet, self).__init__(block = models.resnet.BasicBlock, layers = [3, 4, 6, 3])

        filePath = Path(__file__).parent / 'resnet34-b627a593.pth'
        if filePath.is_file():
#             model_resnet34 = models.resnet34()
#             model_resnet34.load_state_dict(torch.load('./data/resnet34-b627a593.pth'))
            pass
        else:
            download_url_to_file(url='https://download.pytorch.org/models/resnet34-b627a593.pth',file_name='resnet34-b627a593.pth', progress=True)

        self.load_state_dict(torch.load(filePath))

        self.avgpool = nn.AvgPool2d((3, 3))
        self.AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d((1, 1))
        #10 10 10
        #20 18 17
        a, b, c = para
        self.layer5= nn.Sequential(
            nn.ConvTranspose2d(in_channels = 512, out_channels = 128, kernel_size=a,  padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.layer6= nn.Sequential(
            nn.ConvTranspose2d(in_channels = 128, out_channels = 16, kernel_size=b, stride = 2, padding=0, bias=False),
            nn.BatchNorm2d(16),#21
            nn.ReLU(inplace=True)
        )
        
        self.layer7= nn.Sequential(
            nn.ConvTranspose2d(16, 1, kernel_size=c, stride = 2, bias=False),
            nn.ReLU()
        )
        
        
        in_channels=3
        self.conv_first = nn.Conv2d(in_channels, out_channels=64, kernel_size=3, stride=2, padding=3, bias=False)
        
            
    def forward(self,x):
        
        x = self.conv_first(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.AdaptiveAvgPool2d(x)
        x = self.layer5(x)#avgpool(x)
        x = self.layer6(x)
        x = self.layer7(x)
        
        return(x)


def train_loop(dataloader, model, loss_fn, optimizer, mask64, device, y_flag, mask_y=None):
    size = len(dataloader.dataset)
    rloss = 0
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        X = X.float()
#         print(X.shape)
        pred = model(X)
        
        pred = pred[:, :, mask64]#torch.squeeze(pred)[mask64]#torch.flatten(torch.mul(model(X), ))
        pred = torch.squeeze(pred)
        
        y = y.float()
        if y_flag == 0:
            y = y[:, torch.flatten(mask_y)]
        
        # print(pred.shape, y.shape)
        
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            rloss = loss
    return(rloss)


def test_loop(dataloader, model, loss_fn, mask64, device, y_flag, mask_y=None):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.float()
            y = y.float()
            
            pred = model(X)
            pred = pred[:,:,mask64]##torch.mul(, mask)
            pred = torch.squeeze(pred)
                    
            y = y.float()
            if y_flag == 0:
                y = y[:, torch.flatten(mask_y)]
            
            test_loss += loss_fn(pred, y).item()
#             correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
#     correct /= size
    # print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return(test_loss)


def generate_input(protein_table_i, index_x, index_y):
    # df_out_i = run_data.iloc[gene, 2:]
    # assess = str(run_data.iloc[gene, 0]) + '_' + str(run_data.iloc[gene, 1]) + '_norm'
    half_tmpX = np.array(protein_table_i[index_x[0]:index_x[1]])#[::-1]#[::2]#
    half_tmpY = np.array(protein_table_i[index_y[0]:index_y[1]])#[::-1]#[::2]#
    
    
    half_tmpXY_mean = (1e-5+np.mean(np.concatenate([half_tmpX, half_tmpY]))/500)
    half_tmpX /= half_tmpXY_mean
    half_tmpY /= half_tmpXY_mean

    Xrep = np.tile(np.array(half_tmpX)[:,np.newaxis], int(len(half_tmpY)))
    Yrep = np.array([half_tmpY] * int(len(half_tmpX)))
    XYrep = Xrep + Yrep

    input_XYrep = np.float32(np.stack((Xrep, Yrep, XYrep), axis=0))
    
    return(input_XYrep, half_tmpXY_mean)
    
    
def generate_val_distribution(model, C3C5_input, index_x, index_y, mask64, device):
    input_XYrep, half_tmpXY_mean = generate_input(C3C5_input, index_x, index_y)
    input_XYrep_tensor = torch.tensor(input_XYrep)
    input_XYrep_tensor = input_XYrep_tensor.to(device)
    
    # print(input_XYrep.shape)
    a,b,c = input_XYrep_tensor.size()
    # pred = model(input_XYrep_tensor.reshape(1, a,b,c))#
    
    pred = model(input_XYrep_tensor.reshape(1, a,b,c))#input_XYrep_tensor.shape()
    pred = pred[:, :, mask64]
    # pred = torch.squeeze(pred)
    # pred[:,:,mask64 == False] = np.nan
    
    tmp_return = pred.squeeze().detach().cpu().numpy() * half_tmpXY_mean #[10:60, 8:82]
    return(tmp_return)