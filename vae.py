import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import lightning.pytorch as pl

import time

class NT_VAE(pl.LightningModule):
    def __init__(self,
                 in_features=5000,
                 latent_dim=128,
                 beta_factor=1e-5,
                 beta_peak_epoch=4,
                 positional_encoding=True,
                 dataset_size=98314,
                 batch_size=128,
                 lr=1e-3, 
                 lr_schedule=[2, 20],
                 weight_decay=1e-5):
        super().__init__()
        self.save_hyperparameters()
        
        embed_dim = 32
        self.embedding = nn.Linear(1, embed_dim)
        self.initial_downsample = nn.Sequential(nn.Linear(in_features, latent_dim*16),
                                                nn.LeakyReLU())
        self.encoder = nn.Sequential(Transformer_VAE_Enc(dim=embed_dim,
                                                        num_heads=4,
                                                        ff_dim=256,
                                                        seq_len_in=latent_dim*16, 
                                                        seq_len_out=latent_dim*8),
                                     Transformer_VAE_Enc(dim=embed_dim,
                                                        num_heads=4,
                                                        ff_dim=256,
                                                        seq_len_in=latent_dim*8, 
                                                        seq_len_out=latent_dim*4),
                                     Transformer_VAE_Enc(dim=embed_dim,
                                                        num_heads=4,
                                                        ff_dim=256,
                                                        seq_len_in=latent_dim*4, 
                                                        seq_len_out=latent_dim*2)
                                     )
        self.latent_output = nn.Linear(embed_dim, 1)
        self.decoder_embedding = nn.Linear(1, embed_dim)
        
        self.tgt_embedding = nn.Parameter(torch.randn(latent_dim, embed_dim))
        
        self.decoder = nn.Sequential(Transformer_VAE_Dec(dim=embed_dim,
                                                        num_heads=4,
                                                        ff_dim=256,
                                                        seq_len_in=latent_dim, 
                                                        seq_len_out=latent_dim*4),
                                     Transformer_VAE_Dec(dim=embed_dim,
                                                        num_heads=4,
                                                        ff_dim=256,
                                                        seq_len_in=latent_dim*4, 
                                                        seq_len_out=latent_dim*16),
                                     Transformer_VAE_Dec(dim=embed_dim,
                                                        num_heads=4,
                                                        ff_dim=256,
                                                        seq_len_in=latent_dim*16, 
                                                        seq_len_out=in_features)
                                     )
        self.output = nn.Linear(embed_dim, 1)
        
        self.fc_mu = nn.Linear(latent_dim*2, latent_dim)
        self.fc_logvar = nn.Linear(latent_dim*2, latent_dim)                   
        
        if positional_encoding:
            self.positional_encoding = nn.Linear(3, in_features)
        
        self.beta = 0.
        self.iter = 0
        self.beta_factor = beta_factor
        self.total_steps = dataset_size * beta_peak_epoch

    def encode(self, inputs):
        inputs = self.initial_downsample(inputs)
        h = self.embedding(inputs.unsqueeze(-1))
        h = self.encoder(h)
        h = self.latent_output(h).squeeze()
        
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        z = self.decoder_embedding(z.unsqueeze(-1))
        tgt = self.tgt_embedding.unsqueeze(0).expand(z.size(0), -1, -1)
        for layer in self.decoder:
            z = layer(z, tgt)
        outputs = self.output(z)
        return outputs.squeeze()
    
    def forward(self, inputs, pos=None):
        if self.hparams.positional_encoding:
            inputs = inputs + self.positional_encoding(pos)
        mu, logvar = self.encode(inputs)
        z = self.reparameterize(mu, logvar)
        outputs = self.decode(z)
        outputs = torch.softmax(outputs, dim=-1)
        return outputs, mu, logvar

    def kl_divergence(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    def training_step(self, batch, batch_idx):
        if self.hparams.positional_encoding:
            inputs, pos = batch
            outputs, mu, logvar = self(inputs, pos)
        else:
            inputs = batch
            outputs, mu, logvar = self(inputs)
            
        reconstruction_loss = nll_poisson_loss(inputs, outputs)
        # reconstruction_loss = F.mse_loss(inputs, outputs)
        kl_loss = self.kl_divergence(mu, logvar)
        
        loss = reconstruction_loss + (self.beta*kl_loss)
        
        # cosine annealing for beta term 
        self.beta = self.beta_factor * ((np.cos(np.pi * (self.iter / self.total_steps - 1)) + 1) / 2)
        self.iter += 1
        
        self.log("train_loss", loss, batch_size=self.hparams.batch_size)
        self.log("kl_loss", kl_loss, batch_size=self.hparams.batch_size)
        self.log("reco_loss", reconstruction_loss, batch_size=self.hparams.batch_size)
        self.log("beta", self.beta, batch_size=self.hparams.batch_size)
        return loss
    
    def validation_step(self, batch, batch_idx):
        if self.hparams.positional_encoding:
            inputs, pos = batch
            outputs, mu, logvar = self(inputs, pos)
        else:
            inputs = batch
            outputs, mu, logvar = self(inputs)

        reconstruction_loss = nll_poisson_loss(inputs, outputs)
        # reconstruction_loss = F.mse_loss(inputs, outputs)
        kl_loss = self.kl_divergence(mu, logvar)
        
        loss = reconstruction_loss + (self.beta*kl_loss)
        
        self.log("val_train_loss", loss, batch_size=self.hparams.batch_size)
        self.log("val_kl_loss", kl_loss, batch_size=self.hparams.batch_size)
        self.log("val_reco_loss", reconstruction_loss, batch_size=self.hparams.batch_size)
        return loss
    
    def test_step(self, batch, batch_idx):
        if self.hparams.positional_encoding:
            inputs, pos = batch
            outputs, mu, logvar = self(inputs, pos)
        else:
            inputs = batch
            outputs, mu, logvar = self(inputs)

        reconstruction_loss = nll_poisson_loss(inputs, outputs)
        kl_loss = self.kl_divergence(mu, logvar)
        loss = reconstruction_loss + (self.beta*kl_loss)
        
        np.save(f"./results/outputs_{time.time()}.npy", outputs.cpu().detach().numpy())
        np.save(f"./results/inputs_{time.time()}.npy", inputs.cpu().detach().numpy())
        import pdb; pdb.set_trace()
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, self.hparams.lr_schedule, gamma=0.1)
        return [optimizer], [scheduler]

@torch.compile
class Transformer_VAE_Enc(nn.Module):
    def __init__(self, dim=256,
                 num_heads=8,
                 ff_dim=2048,
                 dropout=0.1,
                 seq_len_in=5000,
                 seq_len_out=1000):
        super(Transformer_VAE_Enc, self).__init__()
        
        self.enc_layer = nn.TransformerEncoderLayer(d_model=dim, 
                                                    nhead=num_heads, 
                                                    dim_feedforward=ff_dim, 
                                                    dropout=dropout,
                                                    batch_first=True)
        self.downsample = nn.Linear(seq_len_in, seq_len_out)
        
    def forward(self, x):
        x = self.enc_layer(x)
        x = x.permute(0, 2, 1)
        x = self.downsample(x)
        x = x.permute(0, 2, 1)
        return x
    
@torch.compile
class Transformer_VAE_Dec(nn.Module):
    def __init__(self, dim=256,
                 num_heads=8,
                 ff_dim=2048,
                 dropout=0.1,
                 seq_len_in=1000,
                 seq_len_out=5000):
        super(Transformer_VAE_Dec, self).__init__()
        
        self.dec_layer = nn.TransformerDecoderLayer(d_model=dim, 
                                                    nhead=num_heads, 
                                                    dim_feedforward=ff_dim, 
                                                    dropout=dropout,
                                                    batch_first=True)
        self.upsample = nn.Linear(seq_len_in, seq_len_out)
        
    def forward(self, x, memory):
        x = self.dec_layer(x, memory)
        x = x.permute(0, 2, 1)
        x = self.upsample(x)
        x = x.permute(0, 2, 1)
        return x

@torch.compile
def nll_poisson_loss(x, x_recon):
    # x: Actual binned counts (integer values)
    # x_recon: Reconstructed probability distribution (values between 0 and 1, summing to 1)
    
    x = torch.exp(x) - 1
    
    # Calculate the rate parameter lambda for each bin
    N = x.sum(dim=-1, keepdim=True)  # Total count per sample
    lambda_ = x_recon * N  # Scale probabilities by total count
    
    # Poisson log-likelihood
    log_factorial_x = torch.lgamma(x + 1)  # log(x!)
    log_likelihood = x * torch.log(lambda_ + 1e-8) - lambda_ - log_factorial_x
    
    # Negative log-likelihood
    nll = -torch.sum(log_likelihood, dim=-1)
    
    return nll.mean()  # Mean over all samples in the batch