# tf.layers.conv2d_transpose(h, 128, 5, strides=2, activation=tf.nn.relu, name="dec_deconv1")
#       h = tf.layers.conv2d_transpose(h, 64, 5, strides=2, activation=tf.nn.relu, name="dec_deconv2")
#       h = tf.layers.conv2d_transpose(h, 32, 6, strides=2, activation=tf.nn.relu, name="dec_deconv3")

import torch
import torch.nn as nn
from constants import LATENT_SIZE

class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape
    def forward(self, x):
        x = x.view(self.shape)
        return x



class VAELin(nn.Module):
    def __init__(self, z_size=LATENT_SIZE, conv_sizes=[4, 8, 16, 32], batch_size=1, learning_rate=0.0001, kl_tolerance=0.5, gpu=False):
        self.z_size=z_size
        self.conv_sizes = conv_sizes # hardcoding to be 4 deep
        self.batch_size=batch_size
        self.learning_rate=learning_rate
        self.kl_tolerance=kl_tolerance
        self.gpu = gpu
        
        super(VAELin, self).__init__()
        
        self.encoder = nn.Sequential(
            Reshape((-1, 3*64*64)),
            nn.Linear(3*64*64, 400),
            nn.ReLU()
        )
        
        self.mu = nn.Linear(400, self.z_size)
        self.logvar = nn.Linear(400, self.z_size)
        
        self.decoder = nn.Sequential(
            nn.Linear(self.z_size, 400),
            nn.ReLU(),
            nn.Linear(400, 3*64*64),
            Reshape((-1, 3,64,64)),
            nn.Sigmoid()
        )
        
        
    def forward(self, original):
        encoded = self.encoder(original)
        mu = self.mu(encoded)
        logvar = self.logvar(encoded)
        sigma = torch.exp(logvar/2.0)
        eps = torch.randn(self.batch_size, self.z_size)
        z = mu + sigma * eps
        decoded = self.decoder(z)

        return original, decoded, encoded, z, mu, logvar
    
    def reconstruction_loss(self, input_imgs, output_imgs):
        return torch.mean(torch.sum((output_imgs - input_imgs)**2, (1, 2, 3)))

    def kl_loss(self, logvar, mu):
        kl_loss = -0.5 * torch.sum(1 + logvar - mu**2 - torch.exp(logvar), 1)
        kl_loss = torch.max(kl_loss, torch.FloatTensor([self.kl_tolerance*self.z_size]).expand_as(kl_loss))
        kl_loss = torch.mean(kl_loss)
        return kl_loss  
    
    def loss(self, original, decoded, mu, logvar):
        return self.kl_loss(logvar, mu) + self.reconstruction_loss(original, decoded)
    