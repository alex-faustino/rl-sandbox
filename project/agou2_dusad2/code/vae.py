# tf.layers.conv2d_transpose(h, 128, 5, strides=2, activation=tf.nn.relu, name="dec_deconv1")
#       h = tf.layers.conv2d_transpose(h, 64, 5, strides=2, activation=tf.nn.relu, name="dec_deconv2")
#       h = tf.layers.conv2d_transpose(h, 32, 6, strides=2, activation=tf.nn.relu, name="dec_deconv3")

import torch
import torch.nn as nn

class Reshape(nn.Module):
    def __init__(self):
        super(Reshape, self).__init__()
    def forward(self, x):
        x = x.view(-1,  4*256, 1, 1)
        return x

    

class VAE(nn.Module):
    def __init__(self, z_size=32, batch_size=1, learning_rate=0.0001, kl_tolerance=0.5, gpu=False):
        self.z_size=z_size
        self.batch_size=batch_size
        self.learning_rate=learning_rate
        self.kl_tolerance=kl_tolerance
        self.gpu = gpu
        
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.ReLU(),
        )
        
        self.mu = nn.Linear(256*2*2, 32)
        self.logvar = nn.Linear(256*2*2, 32)
        
        self.decoder = nn.Sequential(
            nn.Linear(self.z_size,4*256),
            Reshape(),
            nn.ConvTranspose2d(4*256, 128, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=6, stride=2),
            nn.Sigmoid()
        )
        
        
    def forward(self, original):
        encoded = self.encoder(original)
        encoded = encoded.view(-1, 256*2*2)
        
        mu = self.mu(encoded)
        logvar = self.logvar(encoded)
        sigma = torch.exp(logvar/2.0)
        eps = torch.randn(self.batch_size, self.z_size)
        z = mu + sigma * eps
        
        decoded = self.decoder(z)
        
        return original, decoded, encoded, z, mu, logvar
    
    def reconstuction_loss(self, input_imgs, output_imgs):
        return torch.mean(torch.sum((output_imgs - input_imgs)**2, (1, 2, 3)))

    def kl_loss(self, logvar, mu):
        kl_loss = -0.5 + torch.sum(1 + logvar - mu**2 - torch.exp(logvar), 1)
        kl_loss = torch.max(kl_loss, torch.FloatTensor([self.kl_tolerance*self.z_size]).expand_as(kl_loss))
        kl_loss = torch.mean(kl_loss)
        return kl_loss  
    
    def loss(self, original, decoded, mu, logvar):
        return self.kl_loss(logvar, mu) + self.reconstuction_loss(original, decoded)
    