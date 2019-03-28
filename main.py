from __future__ import print_function
import os
from shutil import copyfile

import numpy as np
from scipy.misc import imread, imsave, imrotate, imresize
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from torch.nn import functional as F
from torch.utils.data import DataLoader

from dataset_loader import DatasetFromNPZ
from model import BetaVAE



def train(epoch, model, optimizer, train_loader, beta, device):
    model.train()
    train_loss, bce_x_loss, kld_loss = 0., 0., 0.
    # z_map = np.zeros([1, 16])
    # hr_ab  = np.zeros([1, dim_accumulate])
    for _, batch in enumerate(train_loader):
        h_input = batch[0].to(device)#, batch[1].to(device)
        optimizer.zero_grad()
        h_output, mu, logvar, _ = model(h_input)
        loss, bce_x, kld = loss_function(h_output, h_input, mu, logvar, beta)
        loss.backward()
        train_loss += loss.item()
        bce_x_loss += bce_x.item()
        kld_loss   += kld.item()
        optimizer.step()
        # if epoch == 1 or epoch % 1000 == 0:
        #     if device == 'cuda':
        #         z_map = np.vstack([z_map, z.cpu().detach().numpy()])
        #         hr_ab = np.vstack([hr_ab, h_output.cpu().detach().numpy()])
        #     elif device == 'cpu':
        #         z_map = np.vstack([z_map, z.detach().numpy()])
        #         hr_ab = np.vstack([hr_ab, h_output.detach().numpy()])
    train_loss /= len(train_loader.dataset)
    bce_x_loss /= len(train_loader.dataset)
    kld_loss   /= len(train_loader.dataset)
    if epoch % 10 == 0:
        print('===> Epoch: {:4d}  |  total_loss: {:.3f}  |  bce_x_loss: {:.3f}  |  kld_loss: {:6.4f}'
        .format(epoch, train_loss, bce_x_loss, kld_loss))
    return train_loss, bce_x_loss, kld_loss


def test(epoch, model, test_loader, beta, device):
    model.eval()
    test_loss, bce_x_loss, kld_loss = 0., 0., 0.
    with torch.no_grad():
        for batch in test_loader:
            h_input = batch[0].to(device), batch[1].to(device)
            h_output, mu, logvar, z_map = model(h_input)
            loss, bce_x, kld = loss_function(h_output, h_input, mu, logvar, beta)
            test_loss  += loss.item()
            bce_x_loss += bce_x.item()
            kld_loss   += kld.item()
    test_loss  /= len(test_loader.dataset)
    bce_x_loss /= len(test_loader.dataset)
    kld_loss   /= len(test_loader.dataset)
    # print("===> Epoch {}, total loss: {:.4f}".format(epoch, test_loss))
    return h_output, z_map


def cross_entropy_one_hot(input, target):       #  one-hot encoded vector
    print(target.max(dim=0))
    _, labels = target.max(dim=0)
    return nn.CrossEntropyLoss()(input, labels)


def loss_function(recon_x, x, mu, logvar, beta):     # Reconstruction + KL divergence losses summed over all elements and batch
    BCE_x = F.binary_cross_entropy(recon_x, x, size_average=False)      # BCE = F.binary_cross_entropy(recon_x, x, reduction='sum') # new version of pytorch    # F.binary_cross_entropy(output, target)

    """ Kingma and Welling. Auto-Encoding Variational Bayes. (ICLR, 2014, see Appendix B from VAE paper) """
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    """ Model proposed in original beta-VAE paper (Higgins et al, ICLR, 2017) """
    # beta_vae_loss = recon_loss + self.beta * total_kld
    # beta, default=4, type=float, help='beta parameter for KL-term in original beta-VAE'
    KLD *= beta

    """ Model proposed in understanding beta-VAE paper (Burgess et al, arxiv:1804.03599, 2018) """
    # C = torch.clamp( self.C_max/self.C_stop_iter * self.global_iter, 0, self.C_max.data[0] )
    # beta_vae_loss = recon_loss + self.gamma * (total_kld-C).abs()
    # gamma,       default=500, ype=foat, help='gamma parameter for KL-term in understanding beta-VAE'
    # C_max,       default=25,   type=float, help='capacity parameter(C) of bottleneck channel'
    # C_stop_iter, default=1e5,  type=float, help='when to stop increasing the capacity'
    return BCE_x + KLD, BCE_x, KLD/beta      # MSE_c is dummy here


def data_loader(path_load_npz, batch_size):
    dataset = DatasetFromNPZ(path_load_npz)
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(dataset=dataset, batch_size=len(dataset), shuffle=False, num_workers=4)
    return train_loader, test_loader


def makepath(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def saveCode(dir_save_code):
    copyfile('./main_bvae.py', dir_save_code + 'main_bvae.py')
    copyfile('./model.py', dir_save_code + 'model.py')

    
def saveModel(model, path_save_model):
    torch.save(model, path_save_model)
    print('Checkpoint saved to {}'.format(path_save_model))


def saveNPZ(path_save_npz, beta, n_epochs, batch_size, learning_rate, dim_input, dim_latent, total_loss, BCE_x_loss, KLD_loss, n_checks):
    # layers = '{} | 512 | 128 | 32 | {} | 32 | 128 | 512 | {}'.format(dim_input[0], dim_latent, dim_input[0])
    np.savez( path_save_npz, 
              beta=beta,
              n_epochs=n_epochs, 
              batch_size=batch_size, 
              learning_rate=learning_rate,
              dim_input=dim_input,
              dim_latent=dim_latent,
              total_loss=total_loss,
              BCE_x_loss=BCE_x_loss,
              KLD_loss=KLD_loss,
              n_checks=n_checks)
    print('Parameters saved to {}\n'.format(path_save_npz))
    print('Epochs: {}\nbeta: {}\nBatch Size: {}\nLearning Rate: {}\nInput Dim: {}\nLatent Dim: {}\ntotal_loss: {}\nBCE_x_loss: {}\nKLD_loss: {}\n'
          .format(n_epochs, beta, batch_size, learning_rate, dim_input, dim_latent, total_loss[n_epochs-1], BCE_x_loss[n_epochs-1], KLD_loss[n_epochs-1]))


def gridCheckpoints(n_checks, n_epochs, n_power):
    grid = (np.arange(n_checks))**n_power
    x = n_epochs * grid / grid.max() 
    return [int(i) for i in x]      # dx = [x[idx+1]-i for idx, i in enumerate(x[:-1])]


def saveCheckpoint(dir_checkpoint, epoch, z_radius, z_theta, z_map, h_output):
    np.savez( dir_checkpoint + 'checkpoint-{}'.format(epoch), z_radius=z_radius, z_theta=z_theta, z_map=z_map, h_output=h_output)
    print('===> Checkpoint {} Saved'.format(epoch))


def settings():
    device = "cuda" if torch.cuda.is_available() else "cpu"                     # exp_A if ture else exp_B
    path_load_npz  = './data/results/dataset.npz'
    path_save_npz  = './data/results/nn_parameters.npz'
    path_save_pth  = './data/results/model.pth'
    dir_save_code  = makepath('./data/code/')
    dir_checkpoint = makepath('./data/checkpoint/npz/')
    return device, path_load_npz, path_save_npz, path_save_pth, dir_save_code, dir_checkpoint



''' ############################################################################################## '''

def parameters():
    n_checks   = 50                         # total checkpoints (grid checkpoint)
    n_power    = 3                          # the power of 1 or 3 (grid checkpoint)
    dim_latent = 16                         # total latent variables
    dim_input  = (26496)                    # both input and output size of the sum of real and imag
    dim_accumulate = 26496
    return n_checks, n_power, dim_latent, dim_input, dim_accumulate

def hyperparameters():
    beta          = 1                       # modify the weight of KLD_loss in all R, G, B channel
    n_epochs      = 10000                   # epoch
    batch_size    = 50                      # batch size
    learning_rate = 1e-4                    # 1e-3, large learning rate has high jumps
    return beta, n_epochs, batch_size, learning_rate

''' ############################################################################################## '''



if __name__ == "__main__":

    torch.manual_seed(7)

    print('===> Loading Parameters, Hyperparameters, Paths')
    n_checks, n_power, dim_latent, dim_input, dim_accumulate = parameters()
    beta, n_epochs, batch_size, learning_rate = hyperparameters()
    device, path_load_npz, path_save_npz, path_save_pth, dir_save_code, dir_checkpoint = settings()

    print('===> Loading Dataset (2 x 3 Channels)')
    train_loader, test_loader = data_loader(path_load_npz, batch_size)

    print('===> Save Python Codes')
    saveCode(dir_save_code)

    print('===> Declare Arrays')
    total_loss = np.zeros(n_epochs)
    BCE_x_loss = np.zeros(n_epochs)
    KLD_loss   = np.zeros(n_epochs)

    print('===> Grid of Checkpoints')
    grid_checkpoint = gridCheckpoints(n_checks, n_epochs, n_power)

    print('===> Declare Model (BetaVAE)')
    model = BetaVAE(dim_latent, dim_input).to(device)

    print('===> Declare Optimiser')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print('===> Start Training ...')
    for epoch in range(1, n_epochs + 1):
        total_loss[epoch-1], BCE_x_loss[epoch-1], KLD_loss[epoch-1] = train(epoch, model, optimizer, train_loader, beta, device)
        # if epoch in grid_checkpoint:
        #     h_output, z_map = test(epoch, model, test_loader, beta, device)
        #     z_theta = z_radius
        #     saveCheckpoint(dir_checkpoint, epoch, z_radius, z_theta, z_map, h_output)

    print('===> Saving Files')
    saveModel(model, path_save_pth)
    saveNPZ(path_save_npz, beta, n_epochs, batch_size, learning_rate, dim_input, dim_latent, total_loss, BCE_x_loss, KLD_loss, n_checks)

