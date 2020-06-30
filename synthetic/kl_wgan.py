from collections import OrderedDict

import numpy as np
import torch
import torch.distributions as td
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import tqdm
from numpy.random import randn
from scipy.stats import gaussian_kde
from sklearn.metrics.pairwise import euclidean_distances

import datasets as D

noise_dim = 2
sn = nn.utils.spectral_norm


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 2)
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            sn(nn.Linear(2, 100)),
            nn.LeakyReLU(),
            sn(nn.Linear(100, 100)),
            nn.LeakyReLU(),
            sn(nn.Linear(100, 1))
        )

    def forward(self, x):
        return self.net.forward(x)


def mmd(X, Y):
    dxy = euclidean_distances(X, Y) / X.shape[1]
    dxx = euclidean_distances(X, X) / X.shape[1]
    dyy = euclidean_distances(Y, Y) / X.shape[1]
    kxy = np.exp(-dxy ** 2)
    kxx = np.exp(-dxx ** 2)
    kyy = np.exp(-dyy ** 2)
    return kxx.mean() + kyy.mean() - 2 * kxy.mean()


def kde(X, Y):
    qx = gaussian_kde(X.T, bw_method='scott')
    return qx.logpdf(Y.T).mean()


name_to_dataset = OrderedDict([
    ('grid', D.Grid),
    ('banana', D.Banana),
    ('rings', D.Ring),
    ('uniform', D.Uniform),
    ('cosine', D.Cosine),
    ('funnel', D.Funnel),
    ('multiring', D.Multiring)
])


def hinge_dis(dis_fake, dis_real):
    loss_real = torch.mean(F.relu(1. - dis_real))
    loss_fake = torch.mean(F.relu(1. + dis_fake))
    return loss_real, loss_fake


def hinge_gen(dis_fake):
    loss = -torch.mean(dis_fake)
    return loss


def kl_dis(dis_fake, dis_real):
    loss_real = torch.mean(F.relu(1. - dis_real))
    dis_fake_norm = torch.exp(dis_fake).mean()
    dis_fake_ratio = torch.exp(dis_fake) / dis_fake_norm
    dis_fake = dis_fake * dis_fake_ratio
    loss_fake = torch.mean(F.relu(1. + dis_fake))
    return loss_real, loss_fake


def kl_gen(dis_fake):
    dis_fake_norm = torch.exp(dis_fake).mean()
    dis_fake_ratio = torch.exp(dis_fake) / dis_fake_norm
    dis_fake = dis_fake * dis_fake_ratio
    loss = -torch.mean(dis_fake)
    return loss


def kl_r(dis_fake):
    dis_fake_norm = torch.exp(dis_fake).mean()
    dis_fake_ratio = torch.exp(dis_fake) / dis_fake_norm
    return (dis_fake_ratio / dis_fake_norm) * (dis_fake - torch.logsumexp(dis_fake, dim=0) + torch.log(torch.tensor(dis_fake.size(0)).float()).to('cuda'))


def train_network(name, num_epochs=500, loss_type='hinge', device='cuda', save_id=0):
    print(f'Running {name}, {loss_type}')
    g_net = Generator().to(device)
    g_optim = optim.RMSprop(g_net.parameters(), lr=0.0002)
    d_net = Discriminator().to(device)
    d_optim = optim.RMSprop(d_net.parameters(), lr=0.0002)

    dataset = name_to_dataset[name]()

    train_samples = dataset.sample(5000)
    train_loader = data.DataLoader(data.TensorDataset(
        torch.from_numpy(train_samples).float()), batch_size=256)

    val_samples = dataset.sample(5000)
    # val_loader = data.DataLoader(data.TensorDataset(
    #     torch.from_numpy(val_samples).float()), batch_size=256)

    obj_values = []

    for e in tqdm.tqdm(range(num_epochs)):
        epoch_values = []
        for x in train_loader:
            x = x[0].to(device)
            bs = x.size(0)

            for i in range(5):
                z = torch.randn(bs, noise_dim).to(device)
                d_optim.zero_grad()
                with torch.no_grad():
                    G_x = g_net(z)
                dis_real = d_net(x)
                dis_fake = d_net(G_x)
                if loss_type == 'kl':
                    d_loss_real, d_loss_fake = kl_dis(dis_fake, dis_real)
                elif loss_type == 'hinge':
                    d_loss_real, d_loss_fake = hinge_dis(dis_fake, dis_real)
                else:
                    raise ValueError('loss_type')
                d_loss = d_loss_fake + d_loss_real
                d_loss = d_loss.mean()
                d_loss.backward()
                d_optim.step()

            g_optim.zero_grad()
            z = torch.randn(bs, noise_dim).to(device)
            G_x = g_net(z)
            dis_fake = d_net(G_x)
            if loss_type == 'kl':
                g_loss = kl_gen(dis_fake)
            elif loss_type == 'hinge':
                g_loss = hinge_gen(dis_fake)
            else:
                raise ValueError('loss_type')
            g_loss = g_loss.mean()
            g_loss.backward()
            g_optim.step()

            kl = (kl_r(dis_fake).mean() +
                  kl_gen(dis_fake).mean() + dis_real.mean()).item()
            hinge = (dis_real - dis_fake).mean().item()
            print(kl, hinge)
            if loss_type == 'kl':
                epoch_values.append(kl)
            else:
                epoch_values.append(hinge)

        obj_values.append(np.mean(epoch_values))
        with torch.no_grad():
            z = torch.randn(5000, noise_dim).to('cuda')
            G_x = g_net(z).cpu().numpy()
        print(obj_values[-1], kde(train_samples,
                                  val_samples) - kde(G_x, val_samples))

    with torch.no_grad():
        z = torch.randn(5000, noise_dim).to('cuda')
        G_x = g_net(z).cpu().numpy()

    save_name = f'save_{save_id}/{name}_{loss_type}.npy'
    np.save(save_name, G_x)

    obj_values = np.array(obj_values)
    save_name = f'save_{save_id}/{name}_{loss_type}.obj.npy'
    np.save(save_name, obj_values)


if __name__ == '__main__':
    for name in ['uniform', 'funnel', 'rings']:
        for save_id in [0]:
            for loss_type in ['kl']:
                train_network(name, num_epochs=500,
                              loss_type=loss_type, save_id=save_id)
