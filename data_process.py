import os
from os import listdir
from os.path import join

import numpy as np
from numpy.fft import fft, ifft, fftshift
from scipy.misc import imread, imsave, imrotate, imresize

import scipy.io.wavfile as wave
from scipy.io import wavfile
from scipy import signal

from matplotlib import pyplot as plt

import torch
from torchvision.utils import save_image



def normalise(arr):
    return (arr - arr.min())/(arr - arr.min()).max()


def parameters_normalise(z):
    re_min = np.real(z).min()
    im_min = np.imag(z).min()  
    re_max = np.real(z).max()
    im_max = np.imag(z).max()
    return re_min, im_min, re_max, im_max


def normalise_cpx(z, par_margin):
    re_min, im_min, re_max, im_max = par_margin
    re_dif = re_max - re_min
    im_dif = im_max - im_min
    z_normalised = (np.real(z) - re_min)/re_dif + 1j*(np.imag(z) - im_min)/im_dif
    return z_normalised


def denormalise_cpx(z, par_margin):
    re_min, im_min, re_max, im_max = par_margin
    re_dif = re_max - re_min
    im_dif = im_max - im_min
    z_denormalised = (np.real(z)*re_dif + re_min) + 1j*((np.imag(z)*im_dif + im_min))
    return z_denormalised


def fil(X_wav, Y_wav):
    return ifft( Y_wav / X_wav )



rate, data0 = wave.read('./data/ow.wav')
rate, data1 = wave.read('./data/how.wav')
rate, data2 = wave.read('./data/are.wav')
rate, data3 = wave.read('./data/you.wav')
print(data0.shape, data1.shape, data2.shape, data3.shape)

data0_ = data0[(13824-13248)//2:(13824-13248)//2+13248]
data1_ = data1[(14976-13248)//2:(14976-13248)//2+13248]
data2_ = data2.copy()
data3_ = data3[(14400-13248)//2:(14400-13248)//2+13248]

wavfile.write('./data/0.wav', rate, data0_)
wavfile.write('./data/1.wav', rate, data1_)
wavfile.write('./data/2.wav', rate, data2_)
wavfile.write('./data/3.wav', rate, data3_)
print(data0_.shape, data1_.shape, data2_.shape, data3_.shape)


rate, data0 = wave.read('./data/0.wav')
rate, data1 = wave.read('./data/1.wav')
rate, data2 = wave.read('./data/2.wav')
rate, data3 = wave.read('./data/3.wav')
print(rate, data0.shape, data1.shape, data2.shape, data3.shape)

X0_cpx = np.fft.fft(data0)
Y0_cpx = np.zeros([3, 13248], dtype=np.complex64)
Y0_cpx[0, :] = np.fft.fft(data1)
Y0_cpx[1, :] = np.fft.fft(data2)
Y0_cpx[2, :] = np.fft.fft(data3)

h0_cpx = np.zeros([3, 13248], dtype=np.complex64)
par_nor0 = np.zeros([3, 4], dtype=np.float32)
par_margin0 = np.zeros([4], dtype=np.float32) 
h0_cpx_nor = np.zeros([3, 13248], dtype=np.complex64)
h0_cpx_denor = np.zeros([3, 13248], dtype=np.complex64)
h0_2ch_nor = np.zeros([3, 2, 13248], dtype=np.float32)
yr0 = np.zeros([3, 13248], dtype=np.float32)

for i in range(3):
    h0_cpx[i, :] = fil( X0_cpx, Y0_cpx[i, :] )

for i in range(3):
    par_nor0[i, 0], par_nor0[i, 1], par_nor0[i, 2], par_nor0[i, 3] = parameters_normalise(h0_cpx[i, :])

par_margin0 = par_nor0[:, 0].min(), par_nor0[:, 1].min(), par_nor0[:, 2].max(), par_nor0[:, 3].max()

for i in range(3):
    h0_cpx_nor[i, :] = normalise_cpx(h0_cpx[i, :], par_margin0)

for i in range(3):
    h0_2ch_nor[i, 0, :] = np.real(h0_cpx_nor[i, :])
    h0_2ch_nor[i, 1, :] = np.imag(h0_cpx_nor[i, :])

for i in range(3):
    # h0_cpx_denor[i, :] = denormalise_cpx( h0_cpx_nor[i, :], par_margin0)
    h0_cpx_denor[i, :] = denormalise_cpx( (h0_2ch_nor[i, 0, :] + 1j*h0_2ch_nor[i, 1, :]), par_margin0)

for i in range(3):
    yr0[i, :] = np.real( ifft(X0_cpx * fft(h0_cpx_denor[i, :])) )


np.savez('./data/dataset.npz', X0_cpx=X0_cpx, Y0_cpx=Y0_cpx, h0_2ch_nor=h0_2ch_nor, par_margin0=par_margin0)


# spectrum = np.fft.fft(data)
# mod = np.round(np.real(np.fft.ifft(spectrum)))      # important: round
# mod = np.array([np.int16(i) for i in mod])          # important: int16


# mod = np.hstack([data1, data2, data3])
# wavfile.write('./data/modified.wav', rate, mod)


time = data0.size/rate
t_axis = np.linspace(0, time, data0.shape[0]//20+1)

print('Data:', data0)
print('Sampling rate:', rate)
print('Audio length:', time, 'seconds')
print('Lowest amplitude:', min(data0))
print('Highest amplitude:', max(data0))

plt.figure()
plt.subplot(221)
plt.plot(t_axis, data0[::20], 'k-', label='audio: ow', linewidth=1)
plt.ylabel('Amplitude (a.u.)')
plt.xlabel('Time (second)')
plt.legend()

plt.subplot(222)
plt.plot(t_axis, data1[::20], 'r-', label='audio: how', linewidth=1)
plt.ylabel('Amplitude (a.u.)')
plt.xlabel('Time (second)')
plt.legend()

plt.subplot(223)
plt.plot(t_axis, data2[::20], 'g-', label='audio: are', linewidth=1)
plt.ylabel('Amplitude (a.u.)')
plt.xlabel('Time (second)')
plt.legend()

plt.subplot(224)
plt.plot(t_axis, data3[::20], 'b-', label='audio: you', linewidth=1)
plt.ylabel('Amplitude (a.u.)')
plt.xlabel('Time (second)')
plt.legend()
plt.show()
