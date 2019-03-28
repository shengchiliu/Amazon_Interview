from __future__ import print_function
import os
import argparse

import numpy as np
from numpy.fft import fft, ifft, fftshift
from scipy.misc import imread, imsave, imrotate, imresize
from scipy import interpolate
from matplotlib import pyplot as plt
from matplotlib import transforms

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils import data
from torch.utils.data import DataLoader 
from torchvision import datasets
from torchvision.utils import save_image
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize

import scipy.io.wavfile as wave
from scipy.io import wavfile
from scipy import signal



def makepath(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


# def normalise(rec):                                         # rec = np.abs(rec)
#     return  (rec - rec.min())/(rec - rec.min()).max()


def normalise_bright(rec):                                  # rec = np.abs(rec)
    rec[np.where(rec > 0.60*rec.max())] = 0.60*rec.max()    # enhance light intensity
    return  (rec - rec.min())/(rec - rec.min()).max()


def normalise_dark(rec):                                    # rec = np.abs(rec)
    rec[np.where(rec > 0.75*rec.max())] = 0.75*rec.max()    # enhance light intensity
    return  (rec - rec.min())/(rec - rec.min()).max()


def colour_rec(dir_load, dir_save, filename):
    yr_r = imread(dir_load + filename + '_R.png', 'L')
    yr_g = imread(dir_load + filename + '_G.png', 'L')
    yr_b = imread(dir_load + filename + '_B.png', 'L')
    yr = np.dstack([yr_r, yr_g, yr_b])
    yr = normalise_dark(yr)
    imsave( dir_save + filename + '.png', yr)


def saveImages(dir_save, dataset, filename, nrow_save_image):
    len_samples, _, dim_size = dataset.shape
    n = len_samples//6

    # Declare Arrays
    yr0 = np.zeros([n, 1, dim_size])
    yr1 = np.zeros([n, 1, dim_size])
    yr2 = np.zeros([n, 1, dim_size])
    yr3 = np.zeros([n, 1, dim_size])
    yr4 = np.zeros([n, 1, dim_size])
    yr5 = np.zeros([n, 1, dim_size])

    # # Modify
    # dataset[n-2+0*n,0,:,:] = dataset[n-3+0*n,0,:,:]
    # dataset[n-2+1*n,0,:,:] = dataset[n-3+1*n,0,:,:]
    # dataset[n-2+2*n,0,:,:] = dataset[n-3+2*n,0,:,:]
    # dataset[  0+3*n,0,:,:] = dataset[  2+3*n,0,:,:]
    # dataset[  0+4*n,0,:,:] = dataset[  2+4*n,0,:,:]
    # dataset[  0+5*n,0,:,:] = dataset[  2+5*n,0,:,:]
    # dataset[n-1+0*n,0,:,:] = dataset[n-3+0*n,0,:,:]
    # dataset[n-1+1*n,0,:,:] = dataset[n-3+1*n,0,:,:]
    # dataset[n-1+2*n,0,:,:] = dataset[n-3+2*n,0,:,:]
    # dataset[  1+3*n,0,:,:] = dataset[  2+3*n,0,:,:]
    # dataset[  1+4*n,0,:,:] = dataset[  2+4*n,0,:,:]
    # dataset[  1+5*n,0,:,:] = dataset[  2+5*n,0,:,:]

    # Save Single Image 1-by-1
    for i in range(n):
        yr0[i,0,:,:] = dataset[i+0*n,0,:,:]
        yr1[i,0,:,:] = dataset[i+1*n,0,:,:]
        yr2[i,0,:,:] = dataset[i+2*n,0,:,:]
        yr3[i,0,:,:] = dataset[i+3*n,0,:,:]
        yr4[i,0,:,:] = dataset[i+4*n,0,:,:]
        yr5[i,0,:,:] = dataset[i+5*n,0,:,:]
        if filename in ['hr_re', 'hr_im']:
            imsave( dir_save + "{}_a{}.png".format(filename, i), np.dstack([yr0[i,0,:,:], yr1[i,0,:,:], yr2[i,0,:,:]]) )
            imsave( dir_save + "{}_b{}.png".format(filename, i), np.dstack([yr3[i,0,:,:], yr4[i,0,:,:], yr5[i,0,:,:]]) )
        else:
            imsave( dir_save + "{}_a{}.png".format(filename, i), normalise_bright(np.dstack([yr0[i,0,:,:], yr1[i,0,:,:], yr2[i,0,:,:]])) )
            imsave( dir_save + "{}_b{}.png".format(filename, i), normalise_bright(np.dstack([yr3[i,0,:,:], yr4[i,0,:,:], yr5[i,0,:,:]])) )
        print('{}_a{} and {}_b{} are saved'.format(filename, i, filename, i))
    save_image(torch.from_numpy(yr0), dir_save + '{}0.png'.format(filename), nrow=nrow_save_image, padding=2)
    save_image(torch.from_numpy(yr1), dir_save + '{}1.png'.format(filename), nrow=nrow_save_image, padding=2)
    save_image(torch.from_numpy(yr2), dir_save + '{}2.png'.format(filename), nrow=nrow_save_image, padding=2)
    save_image(torch.from_numpy(yr3), dir_save + '{}3.png'.format(filename), nrow=nrow_save_image, padding=2)
    save_image(torch.from_numpy(yr4), dir_save + '{}4.png'.format(filename), nrow=nrow_save_image, padding=2)
    save_image(torch.from_numpy(yr5), dir_save + '{}5.png'.format(filename), nrow=nrow_save_image, padding=2)
    
    # Save Tiled Image
    yr_r = imread(dir_save + '{}0.png'.format(filename), 'L')
    yr_g = imread(dir_save + '{}1.png'.format(filename), 'L')
    yr_b = imread(dir_save + '{}2.png'.format(filename), 'L')
    yr = np.dstack([yr_r, yr_g, yr_b])
    yr = normalise_dark(yr)
    imsave( dir_save + '{}_a.png'.format(filename), yr)
    yr_r = imread(dir_save + '{}3.png'.format(filename), 'L')
    yr_g = imread(dir_save + '{}4.png'.format(filename), 'L')
    yr_b = imread(dir_save + '{}5.png'.format(filename), 'L')
    yr = np.dstack([yr_r, yr_g, yr_b])
    yr = normalise_dark(yr)
    imsave( dir_save + '{}_b.png'.format(filename), yr)


def ShowZLine1D(z, z_sampled):
    len_channel = len(z)//6
    len_sampled = len(z_sampled)//6
    global n_z_line
    n_z_line = 0
    def toggle_images(event):
        global n_z_line
        if event.key == 'right':
            plt.clf()
            n_z_line += 1
        if event.key == 'left':
            plt.clf()
            n_z_line -= 1

        # base = plt.gca().transData                      # rotation plot graph
        # rot = transforms.Affine2D().rotate_deg(90)      # rotation plot graph

        # Reconstructed Z
        plt.subplot(4,6,1)
        plt.imshow(z[len_channel*0:len_channel*1, :])
        plt.title('Z: Ch0')
        plt.subplot(4,6,2)
        plt.imshow(z[len_channel*1:len_channel*2, :])
        plt.title('Z: Ch1')
        plt.subplot(4,6,3)
        plt.imshow(z[len_channel*2:len_channel*3, :])
        plt.title('Z: Ch2')
        plt.subplot(4,6,4)
        plt.imshow(z[len_channel*3:len_channel*4, :])
        plt.title('Z: Ch3')
        plt.subplot(4,6,5)
        plt.imshow(z[len_channel*4:len_channel*5, :])
        plt.title('Z: Ch4')
        plt.subplot(4,6,6)
        plt.imshow(z[len_channel*5:len_channel*6, :])
        plt.title('Z: Ch5')

        # Reconstructed Z in 1D
        plt.subplot(4,6,7)
        plt.plot(z[len_channel*0:len_channel*1, n_z_line], '-r')
        plt.title('Line {}'.format(n_z_line))
        plt.axis([0, len_channel, -5, 5])
        plt.subplot(4,6,8)
        plt.plot(z[len_channel*1:len_channel*2, n_z_line], '-g')
        plt.title('Line {}'.format(n_z_line))
        plt.axis([0, len_channel, -5, 5])
        plt.subplot(4,6,9)
        plt.plot(z[len_channel*2:len_channel*3, n_z_line], '-b')
        plt.title('Line {}'.format(n_z_line))
        plt.axis([0, len_channel, -5, 5])
        plt.subplot(4,6,10)
        plt.plot(z[len_channel*3:len_channel*4, n_z_line], '-r')
        plt.title('Line {}'.format(n_z_line))
        plt.axis([0, len_channel, -5, 5])
        plt.subplot(4,6,11)
        plt.plot(z[len_channel*4:len_channel*5, n_z_line], '-g')
        plt.title('Line {}'.format(n_z_line))
        plt.axis([0, len_channel, -5, 5])
        plt.subplot(4,6,12)
        plt.plot(z[len_channel*5:len_channel*6, n_z_line], '-b')
        plt.title('Line {}'.format(n_z_line))
        plt.axis([0, len_channel, -5, 5])

        # Sampled Z
        plt.subplot(4,6,13)
        plt.imshow(z_sampled[len_sampled*0:len_sampled*1, :])
        plt.title('Z Sampled: Ch0')
        plt.subplot(4,6,14)
        plt.imshow(z_sampled[len_sampled*1:len_sampled*2, :])
        plt.title('Z Sampled: Ch1')
        plt.subplot(4,6,15)
        plt.imshow(z_sampled[len_sampled*2:len_sampled*3, :])
        plt.title('Z Sampled: Ch2')
        plt.subplot(4,6,16)
        plt.imshow(z_sampled[len_sampled*3:len_sampled*4, :])
        plt.title('Z Sampled: Ch3')
        plt.subplot(4,6,17)
        plt.imshow(z_sampled[len_sampled*4:len_sampled*5, :])
        plt.title('Z Sampled: Ch4')
        plt.subplot(4,6,18)
        plt.imshow(z_sampled[len_sampled*5:len_sampled*6, :])
        plt.title('Z Sampled: Ch5')

        # Sampled Z in 1D
        plt.subplot(4,6,19)
        plt.plot(z_sampled[len_sampled*0:len_sampled*1, n_z_line], '-r')
        plt.title('Line {}'.format(n_z_line))
        plt.axis([0, len_sampled, -5, 5])
        plt.subplot(4,6,20)
        plt.plot(z_sampled[len_sampled*1:len_sampled*2, n_z_line], '-g')
        plt.title('Line {}'.format(n_z_line))
        plt.axis([0, len_sampled, -5, 5])
        plt.subplot(4,6,21)
        plt.plot(z_sampled[len_sampled*2:len_sampled*3, n_z_line], '-b')
        plt.title('Line {}'.format(n_z_line))
        plt.axis([0, len_sampled, -5, 5])
        plt.subplot(4,6,22)
        plt.plot(z_sampled[len_sampled*3:len_sampled*4, n_z_line], '-r')
        plt.title('Line {}'.format(n_z_line))
        plt.axis([0, len_sampled, -5, 5])
        plt.subplot(4,6,23)
        plt.plot(z_sampled[len_sampled*4:len_sampled*5, n_z_line], '-g')
        plt.title('Line {}'.format(n_z_line))
        plt.axis([0, len_sampled, -5, 5])
        plt.subplot(4,6,24)
        plt.plot(z_sampled[len_sampled*5:len_sampled*6, n_z_line], '-b')    # plt.plot(z_sampled[len_sampled*5:len_sampled*6, n_z_line], '-b', transform= rot + base)
        plt.title('Line {}'.format(n_z_line))
        plt.axis([0, len_sampled, -5, 5])
        plt.draw()
    plt.figure()
    plt.title('z dim: {}'.format(z.shape))
    plt.connect('key_press_event', toggle_images)
    plt.show()


def zGradientEffect(z, n_slice_z, z_channels):
    z_channel_0, z_channel_1, z_channel_2, z_channel_3, z_channel_4, z_channel_5 = z_channels
    len_dataset, _ = z.shape
    len_channel = len_dataset//6
    n_center = len_channel//2
    len_sample = n_slice_z * len_channel

    centers = np.arange(n_center, n_center + len_channel*6, len_channel)
    z_basis0 = z[centers[0]:centers[0]+1, :]    # basis z combination, center of the dataset    # [lenx1, 16]
    z_basis1 = z[centers[1]:centers[1]+1, :]    # basis z combination, center of the dataset    # [lenx1, 16]
    z_basis2 = z[centers[2]:centers[2]+1, :]    # basis z combination, center of the dataset    # [lenx1, 16]
    z_basis3 = z[centers[3]:centers[3]+1, :]    # basis z combination, center of the dataset    # [lenx1, 16]
    z_basis4 = z[centers[4]:centers[4]+1, :]    # basis z combination, center of the dataset    # [lenx1, 16]
    z_basis5 = z[centers[5]:centers[5]+1, :]    # basis z combination, center of the dataset    # [lenx1, 16]

    z_sampled0 = np.kron(z_basis0, np.ones((len_sample, 1)))                                     # [lenx6, 16]
    z_sampled1 = np.kron(z_basis1, np.ones((len_sample, 1)))                                     # [lenx6, 16]
    z_sampled2 = np.kron(z_basis2, np.ones((len_sample, 1)))                                     # [lenx6, 16]
    z_sampled3 = np.kron(z_basis3, np.ones((len_sample, 1)))                                     # [lenx6, 16]
    z_sampled4 = np.kron(z_basis4, np.ones((len_sample, 1)))                                     # [lenx6, 16]
    z_sampled5 = np.kron(z_basis5, np.ones((len_sample, 1)))                                     # [lenx6, 16]
    z_sampled  = np.vstack([z_sampled0, z_sampled1, z_sampled2, z_sampled3, z_sampled4, z_sampled5])

    z_min0, z_max0 = z[len_channel*0:len_channel*1, z_channel_0].min(), z[len_channel*0:len_channel*1, z_channel_0].max()
    z_min1, z_max1 = z[len_channel*1:len_channel*2, z_channel_1].min(), z[len_channel*1:len_channel*2, z_channel_1].max()
    z_min2, z_max2 = z[len_channel*2:len_channel*3, z_channel_2].min(), z[len_channel*2:len_channel*3, z_channel_2].max()
    z_min3, z_max3 = z[len_channel*3:len_channel*4, z_channel_3].min(), z[len_channel*3:len_channel*4, z_channel_3].max()
    z_min4, z_max4 = z[len_channel*4:len_channel*5, z_channel_4].min(), z[len_channel*4:len_channel*5, z_channel_4].max()
    z_min5, z_max5 = z[len_channel*5:len_channel*6, z_channel_5].min(), z[len_channel*5:len_channel*6, z_channel_5].max()

    z_grid0 = np.linspace(z_min0, z_max0, len_sample)     # min ~ max
    z_grid1 = np.linspace(z_min1, z_max1, len_sample)     # min ~ max
    z_grid2 = np.linspace(z_min2, z_max2, len_sample)     # min ~ max
    z_grid3 = np.linspace(z_min3, z_max3, len_sample)     # min ~ max
    z_grid4 = np.linspace(z_min4, z_max4, len_sample)     # min ~ max
    z_grid5 = np.linspace(z_min5, z_max5, len_sample)     # min ~ max

    z_sampled[len_sample*0:len_sample*1, z_channel_0] = z_grid0
    z_sampled[len_sample*1:len_sample*2, z_channel_1] = z_grid1
    z_sampled[len_sample*2:len_sample*3, z_channel_2] = z_grid2
    z_sampled[len_sample*3:len_sample*4, z_channel_3] = z_grid3
    z_sampled[len_sample*4:len_sample*5, z_channel_4] = z_grid4
    z_sampled[len_sample*5:len_sample*6, z_channel_5] = z_grid5

    z_sampled[len_sample*0:len_sample*1, 0] =  1.0
    z_sampled[len_sample*1:len_sample*2, 0] =  0.6
    z_sampled[len_sample*2:len_sample*3, 0] =  0.2
    z_sampled[len_sample*3:len_sample*4, 0] = -0.2
    z_sampled[len_sample*4:len_sample*5, 0] = -0.6
    z_sampled[len_sample*5:len_sample*6, 0] = -1.0

    print('\nChannel z0 in R, G, B: ({}, {}, {})'.format(z_channel_0, z_channel_1, z_channel_2))
    print('Channel z1 in R, G, B: ({}, {}, {})\n'.format(z_channel_3, z_channel_4, z_channel_5))
    print('z_min0, z_max0: ({:.4f}, {:.4f})\nz_min1, z_max1: ({:.4f}, {:.4f})\nz_min2, z_max2: ({:.4f}, {:.4f})\n\nDim_z: {}\n'
    .format(z_min0, z_max0, z_min1, z_max1, z_min2, z_max2, z_sampled.shape))
    print('z_min3, z_max3: ({:.4f}, {:.4f})\nz_min4, z_max4: ({:.4f}, {:.4f})\nz_min5, z_max5: ({:.4f}, {:.4f})\n\nDim_z: {}\n'
    .format(z_min3, z_max3, z_min4, z_max4, z_min5, z_max5, z_sampled.shape))
    return z_sampled


def zInterpolate(z, n_slice_z):
    len_dataset, dim_latent = z.shape
    x = np.linspace(0, 1, len_dataset)
    xnew = np.linspace(0, 1, len_dataset*n_slice_z)
    z_sampled = np.zeros([len(xnew), dim_latent])
    for i in range(dim_latent):
        f = interpolate.interp1d(x, z[:,i])
        z_sampled[:,i] = f(xnew)
    # if model_type == 'CVAE':
    #     len_samples = len(z_sampled)//6
    #     z_sampled[0*len_samples:1*len_samples, 0] =  1.0
    #     z_sampled[1*len_samples:2*len_samples, 0] =  0.6
    #     z_sampled[2*len_samples:3*len_samples, 0] =  0.2
    #     z_sampled[3*len_samples:4*len_samples, 0] = -0.2
    #     z_sampled[4*len_samples:5*len_samples, 0] = -0.6
    #     z_sampled[5*len_samples:6*len_samples, 0] = -1.0
    # if wrong_sampled is True:
    #     len_samples = len(z_sampled)//6
    #     z_sampled[0*len_samples:1*len_samples, 0] =  1.0
    #     z_sampled[1*len_samples:2*len_samples, 0] =  0.6
    #     z_sampled[2*len_samples:3*len_samples, 0] =  0.2
    #     z_sampled[3*len_samples:4*len_samples, 0] = -0.2
    #     z_sampled[4*len_samples:5*len_samples, 0] = -0.6
    #     z_sampled[5*len_samples:6*len_samples, 0] = -1.0
    return z_sampled


def zInterpolate1ChBVAE(z, n_slice_z, z_channels):
    len_dataset, dim_latent = z.shape
    x = np.linspace(0, 1, len_dataset)
    xnew = np.linspace(0, 1, len_dataset*n_slice_z)
    z_sampled = np.zeros([len(xnew), dim_latent])
    f = interpolate.interp1d(x, z[:, z_channels[0]])
    z_sampled[:,z_channels[0]] = f(xnew)
    return z_sampled


def zInterpolate1ChCVAE(z, n_slice_z, z_channels):
    len_dataset, dim_latent = z.shape
    x = np.linspace(0, 1, len_dataset)
    xnew = np.linspace(0, 1, len_dataset*n_slice_z)
    z_sampled = np.zeros([len(xnew), dim_latent])
    f0 = interpolate.interp1d(x, z[:, 0])
    z_sampled[:,0] = f0(xnew)
    f1 = interpolate.interp1d(x, z[:, z_channels[0]+1])
    z_sampled[:,z_channels[0]+1] = f1(xnew)
    # len_samples = len(z_sampled)//6
    # z_sampled[0*len_samples:1*len_samples, 0] =  1.0
    # z_sampled[1*len_samples:2*len_samples, 0] =  0.6
    # z_sampled[2*len_samples:3*len_samples, 0] =  0.2
    # z_sampled[3*len_samples:4*len_samples, 0] = -0.2
    # z_sampled[4*len_samples:5*len_samples, 0] = -0.6
    # z_sampled[5*len_samples:6*len_samples, 0] = -1.0
    return z_sampled


def denormalise_cpx(z, par_margin):
    re_min, im_min, re_max, im_max = par_margin
    re_dif = re_max - re_min
    im_dif = im_max - im_min
    z_denormalised = (np.real(z)*re_dif + re_min) + 1j*((np.imag(z)*im_dif + im_min))
    return z_denormalised


def reconstruct(X_cpx, par_margin, hr_numpy):
    n = len(hr_numpy)
    dim_size = X_cpx.shape[0]
    Hr = np.zeros([n, dim_size], dtype=np.complex64)
    yr = np.zeros([n, dim_size], dtype=np.float64)
    for i in range(n):
        h_cpx = hr_numpy[i, 0, :] + 1j*hr_numpy[i, 1, :]            # Complex Wavefront Filter (h_cpx)
        h_cpx_denor = denormalise_cpx(h_cpx, par_margin)       # De-normalised Filter (h_cpx_denor)
        H_cpx = fft( h_cpx_denor )                                  # Frequency Domain (H)
        Hr[i, :] = H_cpx
        yr[i, :] = np.real(ifft( X_cpx * H_cpx ))                  # Reconstruction (Inverse FFT)
    return yr, Hr


def sample_z(model, z_sampled, X_cpx, par_margin, dir_save_hr, dir_save_yr):
    model.eval()                                        # model in evaluation mode, not training mode
    dim_size = X_cpx.shape[0]
    z_input = torch.from_numpy( z_sampled ).float()     # MUST use float for transform numpy to torch to sample
    if torch.cuda.is_available():
        z_input = z_input.cuda()
    # if model_type == 'CVAE':
    #     hr_sampled = model.decode(z_input[:,0:1], z_input[:,1:]).cpu()
    # else:
    #     hr_sampled = model.decode(z_input[:,0:1], z_input).cpu()
    hr_sampled = model.decode(z_input).cpu()
    hr_sampled = hr_sampled.view(-1, 2, dim_size)

    # Sampled z
    # imsave(dir_save_z + 'z.png', np.kron(z_input, np.ones((40,40))))

    # Decoded Filters
    # hr_sampled_re = hr_sampled[:,0:1,:]
    # hr_sampled_im = hr_sampled[:,1:2,:]
    # save_image(hr_sampled_re, dir_save_hr + 'hr_real.png', nrow=nrow_save_image, padding=2)
    # save_image(hr_sampled_im, dir_save_hr + 'hr_imag.png', nrow=nrow_save_image, padding=2)
    hr_numpy = hr_sampled.detach().numpy()
    # saveImage1by1( dir_save_hr, hr_numpy )
    
    # Reconstructed Images
    yr_numpy, Hr_numpy = reconstruct(X_cpx, par_margin, hr_numpy)
    # save_image(torch.from_numpy(yr_numpy), dir_save_yr + 'yr.png', nrow=nrow_save_image, padding=2)
    
    return hr_numpy, Hr_numpy, yr_numpy
    


''' ############################################################################################## '''

# Parameters
n_slice_z = 2                                           # setting sampled z; 2, 5
sample_type = 'Interpolate'                             # 'Gradient', 'Interpolate', 'DropoutBVAE', 'DropoutCVAE'
folder  = 'results' #'results-bh-cvae-b2-o300 (gif)'
wrong_sampled = False                                   # show test results if sample wrong z code (see how strong VAE is)
z_channels = [7, 7, 7, 7, 7, 7]

''' ############################################################################################## '''


# Create Paths
dir_load    = './data/{}/'.format(folder)
dir_load_ry = './data/{}/recons_y/'.format(folder)
dir_save_sz = './data/{}/sample_z/'.format(folder)
dir_save_z  = makepath('./data/{}/sample_z/z/'.format(folder))
dir_save_hr = makepath('./data/{}/sample_z/hr/'.format(folder))
dir_save_Hr = makepath('./data/{}/sample_z/Hr/'.format(folder))
dir_save_yr = makepath('./data/{}/sample_z/yr/'.format(folder))


# Load z, X_cpx, Normalisation Parameters
z  = np.load(dir_load_ry + 'analysis_train_loader.npz')['z']
X_cpx = np.load(dir_load + 'dataset.npz')['X0_cpx']
par_margin = np.load(dir_load + 'dataset.npz')['par_margin0']

 
# Sample z Code
z_sampled = zInterpolate(z, n_slice_z)


# Length of Dataset & Size of Maniford
len_dataset = len(z_sampled)


# Load Models
device = "cuda" if torch.cuda.is_available() else "cpu"     # exp_A if ture else exp_B
model = torch.load(dir_load + 'model.pth').to(device)


# Reconstructed Filter & Image from Sampled z
hr, Hr, yr = sample_z(model, z_sampled, X_cpx, par_margin, dir_save_hr, dir_save_yr)
# saveImages(dir_save_hr, hr[:,0:1,:,:], 'hr_re')
# saveImages(dir_save_hr, hr[:,1:2,:,:], 'hr_im')
# saveImages(dir_save_Hr, np.abs(Hr), 'Hr_am')
# saveImages(dir_save_Hr, np.abs(np.angle(Hr)), 'Hr_ph')
# saveImages(dir_save_yr, yr, 'yr', nrow_save_image)    # saveImages(dir_save_yr, yr, 'yr', 25)


# Save .npz File
# z_radius_train = z[:, 0]
# z_radius_test  = z[:, 0]
# z_theta_train  = z[:, 8]
# z_theta_test   = z[:, 8]
# np.savez('./data/{}/sample_z/analysis_sampled_z.npz'.format(folder), 
#          z_sampled=z_sampled, hr=hr, yr=yr,
#          z_radius_train=z_radius_train, z_radius_test=z_radius_test,
#          z_theta_train=z_theta_train, z_theta_test=z_theta_test)


# Show Graphs
# ShowZLine1D(z, z_sampled)


yr = np.round(yr)
yr_mod = np.zeros(yr.shape, dtype=np.int16)

for j in range(len(yr)):
    yr_mod[j, :] = np.array([np.int16(i) for i in yr[j, :]])

wavfile.write('./data/results/modified_1.wav', 24000, yr_mod[0,:])
wavfile.write('./data/results/modified_2.wav', 24000, yr_mod[1,:])
wavfile.write('./data/results/modified_3.wav', 24000, yr_mod[2,:])
wavfile.write('./data/results/modified_4.wav', 24000, yr_mod[3,:])
wavfile.write('./data/results/modified_5.wav', 24000, yr_mod[4,:])
wavfile.write('./data/results/modified_6.wav', 24000, yr_mod[5,:])

