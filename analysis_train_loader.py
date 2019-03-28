from __future__ import print_function
import os
import argparse

import numpy as np
from numpy.fft import fft, ifft, fftshift
from scipy.misc import imread, imsave, imrotate, imresize
from matplotlib import pyplot as plt

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils import data
from torch.utils.data import DataLoader 
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize

from dataset_loader import DatasetFromNPZ
from model import BetaVAE

import scipy.io.wavfile as wave
from scipy.io import wavfile
from scipy import signal



def makepath(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def normalise(arr):
    return (arr - arr.min())/(arr - arr.min()).max()


def normalise_rec(rec):                                     # rec = np.abs(rec)
    rec[np.where(rec > 0.8*rec.max())] = 0.8*rec.max()      # enhance light intensity
    return  (rec - rec.min())/(rec - rec.min()).max()


def denormalise_cpx(z, par_margin):
    re_min, im_min, re_max, im_max = par_margin
    re_dif = re_max - re_min
    im_dif = im_max - im_min
    z_denormalised = (np.real(z)*re_dif + re_min) + 1j*((np.imag(z)*im_dif + im_min))
    return z_denormalised


def saveImage1by1(path, dataset, n_RGB):
    n = len(dataset)
    for i in range(n):
        imsave( path + str('RGB'[n_RGB]) + '-' + str(i) + ".png", dataset[i, 0, :, :])


def saveImgAllTo1(dir_load, path_save, yr, nrow_save_image):
    imgs = np.zeros(yr.shape)
    for i in range(len(yr)):
        img = imread(dir_load + str(i) + ".png", 'L')
        img = normalise(img)
        imgs[i, 0, :, :] = img
    save_image(torch.from_numpy(imgs), path_save, nrow=nrow_save_image, padding=2)


def manifold(arr4D, m, n):
    _, _, h, w = arr4D.shape
    arr2D = np.zeros((h*m, w*n))
    for i in range(m):
        for j in range(n):
            if m < n:
                num = i*(m+1)+j
            elif m > n:
                num = i*(m-1)+j
            else:
                num = i*m+j
            arr4D[num, 0, :, :] = normalise(arr4D[num, 0, :, :])
            arr2D[i*h:(i+1)*h, j*w:(j+1)*w] = arr4D[num, 0, :, :]
    return arr2D


def colour_rec(dir_load, dis_save):
    # Reconstruction & Save for Full-Colour
    yr_r = imread(dir_load + 'yr_R.png', 'L')
    yr_g = imread(dir_load + 'yr_G.png', 'L')
    yr_b = imread(dir_load + 'yr_B.png', 'L')
    yr = np.dstack([yr_r, yr_g, yr_b])
    yr = normalise_rec(yr)
    imsave( dis_save + 'yr.png', yr)


def showTrainingLoss(total_loss, BCE_loss, KLD_loss):
    plt.figure()
    plt.subplot(3,3,1)
    plt.plot(total_loss[:n_epochs[0], 0], 'r')
    plt.title('Total Loss of R')
    plt.subplot(3,3,2)
    plt.plot(BCE_loss[:n_epochs[0], 0], 'r')
    plt.title('BCE Loss of R')
    plt.subplot(3,3,3)
    plt.plot(KLD_loss[:n_epochs[0], 0], 'r')
    plt.title('KLD Loss of R')

    plt.subplot(3,3,4)
    plt.plot(total_loss[:n_epochs[1], 1], 'g')
    plt.title('Total Loss of G')
    plt.subplot(3,3,5)
    plt.plot(BCE_loss[:n_epochs[1], 1], 'g')
    plt.title('BCE Loss of G')
    plt.subplot(3,3,6)
    plt.plot(KLD_loss[:n_epochs[1], 1], 'g')
    plt.title('KLD Loss of G')
    
    plt.subplot(3,3,7)
    plt.plot(total_loss[:n_epochs[2], 2], 'b')
    plt.title('Total Loss of B')
    plt.subplot(3,3,8)
    plt.plot(BCE_loss[:n_epochs[2], 2], 'b')
    plt.title('BCE Loss of B')
    plt.subplot(3,3,9)
    plt.plot(KLD_loss[:n_epochs[2], 2], 'b')
    plt.title('KLD Loss of B')
    plt.show()


def data_loader(path_dataset, n_RGB):
    dataset = DatasetFromNPZ(path_dataset, n_RGB)
    len_dataset = len(dataset)
    train_loader = DataLoader(dataset=dataset, batch_size=len_dataset, shuffle=False, num_workers=4)
    return train_loader, len_dataset


def reconstruct(X_cpx, par_margin, hr_numpy):
    n = len(hr_numpy)
    dim_size = X_cpx.shape[0]
    yr = np.zeros([n, dim_size])
    for i in range(n):
        h_cpx = hr_numpy[i, 0, :] + 1j*hr_numpy[i, 1, :]                  # Complex Wavefront Filter (h_cpx)
        h_cpx_denor = denormalise_cpx(h_cpx, par_margin)              # De-normalised Filter (h_cpx_denor)
        H_cpx = fft( h_cpx_denor )                                             # Frequency Domain (H)
        yr[i, :] = np.real(ifft( X_cpx * H_cpx ))     # Reconstruction (Inverse FFT)
    return yr


def analysis_recon(model, train_loader, dim_input, n_RGB, X_cpx, par_margin, nrow_save_image, dir_z, dir_hr, dir_yr):

    with torch.no_grad():   # Use torch.no_grad() to wrap around the entire inference code. (or use torch.set_grad_enabled）
        # dim_h, dim_w, _ = dim_input
        for _, h_input in enumerate(train_loader):
            pass
        if torch.cuda.is_available():
            h_input = h_input.cuda()
        h_output, _, _, z = model(h_input)
        h_output = h_output.view(-1, 2, dim_input//2)   # 2 channels, real and imag part


        ''' Encoded Representations '''
        # z code in 1D (1 x 16)
        z_numpy = z.cpu().numpy()
        # imsave(dir_z + 'z_{}.png'.format('RGB'[n_RGB]), np.kron(z_numpy, np.ones((40,40))))
        # ShowZLine1D( z_numpy )    # [n,16]

        # z code in 2D (4 x 4)
        # z_2D = z.view(-1, 1, 4, 4)
        # z_2D_numpy = np.kron(z_2D.cpu().numpy(), np.ones((20,20)))
        # z_2D = torch.from_numpy(z_2D_numpy)
        # save_image(z_2D, dir_z + 'z_2D_{}.png'.format('RGB'[n_RGB]), nrow=nrow_save_image, padding=2)
        # saveImage1by1( dir_z, z_2D_numpy )


        ''' Decoded Filters '''
        # h_output_re = h_output[:,0:1,:]
        # h_output_im = h_output[:,1:2,:]
        # save_image(h_output_re, dir_hr + 'hr_real_{}.png'.format('RGB'[n_RGB]), nrow=nrow_save_image, padding=2)
        # save_image(h_output_im, dir_hr + 'hr_imag_{}.png'.format('RGB'[n_RGB]), nrow=nrow_save_image, padding=2)
        hr_numpy = h_output.cpu().numpy()
        # saveImage1by1( dir_hr, hr_numpy, n_RGB )

        
        ''' Reconstructed Images '''
        yr_numpy = reconstruct(X_cpx, par_margin, hr_numpy)
        # save_image(torch.from_numpy(yr_numpy), dir_yr + 'yr_{}.png'.format('RGB'[n_RGB]), nrow=nrow_save_image, padding=2)
        # saveImage1by1( dir_yr, yr_numpy, n_RGB )
        
        # manifold_rec = manifold(yr_numpy, 15, 12) # 9, 8; 15, 12
        # imsave( dir_yr + "manifold_rec.png", manifold_rec)

        return z_numpy, hr_numpy, yr_numpy



""" Load Data """
# Save Images
nrow_save_image = 8

# Set Paths and Directories
dir_train    = './data/results/'
path_dataset = './data/results/dataset.npz'
path_para_nn = './data/results/nn_parameters.npz'

# Build Folders & Paths
dir_ry = makepath('./data/results/recons_y/')
dir_z  = makepath('./data/results/recons_y/z/')
dir_hr = makepath('./data/results/recons_y/hr/')
dir_yr = makepath('./data/results/recons_y/yr/')

# Load Parameters
parameters     = np.load(path_para_nn)
dim_input      = parameters['dim_input']
dim_latent     = parameters['dim_latent']
dim_accumulate = dim_input
print('\nDim of x: {}\nDim of z: {}'.format(dim_input, dim_latent))

# Load Basis Wavefront of the 3D Model at 0 degree
X_cpx = np.load(path_dataset)['X0_cpx']

# Load De-normalisation Parameters
par_margin = np.load(path_dataset)['par_margin0']

# Load Datasets of R, G and B
train_loader, len_dataset = data_loader(path_dataset, 0)

# Load Model
model = torch.load(dir_train + 'model.pth')

# Load Total, BCE and KLD Loss
total_loss = np.load(path_para_nn)['total_loss']
BCE_loss   = np.load(path_para_nn)['BCE_x_loss']
KLD_loss   = np.load(path_para_nn)['KLD_loss']

# Load Epoch Numbers
n_epochs = np.load(path_para_nn)['n_epochs']


""" Analysis of Reconstructed Inputs (imgs, fils, codes, z, h_output, imgs_reconst) """

# Reconstruction & Save for R Channel
z, hr, yr = analysis_recon(model, train_loader, dim_input, 0, X_cpx, par_margin, nrow_save_image, dir_z, dir_hr, dir_yr)


# Save .npz File
np.savez( dir_ry+'analysis_train_loader.npz', z=z, hr=hr, yr=yr)
print('\nanalysisrain_loader.npz saved here ({})'.format(dir_ry))
print('Dim z: {}\nDim hr: {}\nDim yr: {}\n'.format(z.shape, hr.shape, yr.shape))

yr = np.round(yr)
yr_mod = np.zeros(yr.shape, dtype=np.int16)

for j in range(3):
    yr_mod[j, :] = np.array([np.int16(i) for i in yr[j, :]])

wavfile.write('./data/results/modified_1.wav', 24000, yr_mod[0,:])
wavfile.write('./data/results/modified_2.wav', 24000, yr_mod[1,:])
wavfile.write('./data/results/modified_3.wav', 24000, yr_mod[2,:])

