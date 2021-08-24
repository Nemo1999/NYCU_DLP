import torch
import torch.optim as optim
from torchvision.utils import make_grid, save_image
from get_device import device
from evaluator import evaluation_model
from dataset import get_dataloader, unnormalize
<<<<<<< HEAD
from torch.optim.lr_scheduler import StepLR
=======
>>>>>>> ccfb4c8982e1c26908e84288185ac7131ac240aa
from model import Generator, Discriminator, weights_init
from os.path import isfile
from progressbar import progressBar
import numpy as np
import matplotlib.pyplot as plt
import copy

# suppress pytorch warning
import warnings
#from torch.serialization import SourceChangeWarning#, UserWarning
warnings.filterwarnings("ignore")#, category=SourceChangeWarning)


hp = {
    'lr': 0.002,
    'beta1': 0.5,
    'name': 'smooth+crop+smallLR',
    'nz': 100,  # random vector size
    'ngf': 64,  # generator features
    'ndf': 64   # discriminator features
}

<<<<<<< HEAD
MAX_ITER = 100
=======
MAX_ITER = 40
>>>>>>> ccfb4c8982e1c26908e84288185ac7131ac240aa

def hp_2_str(hp):
    return f'lr={hp["lr"]}_beta1={hp["beta1"]}_nz={hp["nz"]}_ngf={hp["ngf"]}_ndf={hp["ndf"]}_{hp["name"]}'

def search_checkpoint(hp):
    name = hp_2_str(hp)
    path = 'checkpoints/'+hp_2_str(hp)+'_ckpt.pth'
    return isfile(path)

def save_checkpoint(hp, ckpt):
    name = hp_2_str(hp)
    path = 'checkpoints/'+hp_2_str(hp)+'_ckpt.pth'
    torch.save(ckpt, path)
<<<<<<< HEAD
    # print(f'checkpoint is save at {path}')

def load_checkpoint(hp, load_type='latest',
                    netG=None, netD=None, optG=None, optD=None, schG=None, schD=None):
=======
    print(f'checkpoint is save at {path}')

def load_checkpoint(hp, load_type='latest',
                    netG=None, netD=None, optG=None, optD=None):
>>>>>>> ccfb4c8982e1c26908e84288185ac7131ac240aa
    name = hp_2_str(hp)
    path = 'checkpoints/'+hp_2_str(hp)+'_ckpt.pth'
    ckpt = torch.load(path)
    if netG:
        netG.load_state_dict(ckpt[load_type]['netG'])
    if netD:
        netD.load_state_dict(ckpt[load_type]['netD'])
    if optG:
        optG.load_state_dict(ckpt[load_type]['optG'])
    if optD:
        optD.load_state_dict(ckpt[load_type]['optD'])
<<<<<<< HEAD
    if schG:
        schG.load_state_dict(ckpt[load_type]['schG'])
    if schD:
        schD.load_state_dict(ckpt[load_type]['schD'])
    return ckpt

def pack_state_dicts(netG, netD, optG, optD, schG=None, schD=None):
=======
    return ckpt

def pack_state_dicts(netG, netD, optG, optD):
>>>>>>> ccfb4c8982e1c26908e84288185ac7131ac240aa
    state = {
        'netG': netG.state_dict(),
        'netD': netD.state_dict(),
        'optG': optG.state_dict(),
<<<<<<< HEAD
        'optD': optD.state_dict(),
        'schG': None,
        'schD': None
    }
    if schG:
        state['schG'] = schG.state_dict(),
    if schD:
        state['schD'] = schD.state_dict(),
    return copy.deepcopy(state)

def train_with_hp(hp):
    dl_train = get_dataloader('train', batch_size=64)
=======
        'optD': optD.state_dict()
    }
    return copy.deepcopy(state)

def train_with_hp(hp):
    dl_train = get_dataloader('train', batch_size=32)
>>>>>>> ccfb4c8982e1c26908e84288185ac7131ac240aa
    dl_test = get_dataloader('eval', batch_size=16)

    nz = hp["nz"]
    ngf = hp["ngf"]
    ndf = hp["ndf"]
    lr = hp["lr"]
    beta1 = hp["beta1"]

    netG = Generator(nz, ngf).to(device)
    netD = Discriminator(ndf).to(device)
    netG.apply(weights_init)
    netD.apply(weights_init)

    optG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
<<<<<<< HEAD
    optD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    if 'sch' in hp and hp['sch']:
        # scheduler = ReduceLROnPlateau(optimizerD, factor=0.5, patience=2)
        schG = StepLR(optG, gamma=0.3, step_size=20)
        schD = StepLR(optD, gamma=0.3, step_size=20)
    else:
        schG = None
        schD = None
=======
    optD = optim.Adam(netD.parameters(), lr=lr/4, betas=(beta1, 0.999))
>>>>>>> ccfb4c8982e1c26908e84288185ac7131ac240aa

    ev = evaluation_model()

    if search_checkpoint(hp):
<<<<<<< HEAD
        ckpt = load_checkpoint(hp, 'latest', netG, netD, optG, optD, schG, schD)
=======
        ckpt = load_checkpoint(hp, 'latest', netG, netD, optG, optD)
>>>>>>> ccfb4c8982e1c26908e84288185ac7131ac240aa
        history = ckpt['history']
    else:
        history = {
            'ep_cnt': -1,
            'best_score': 0.0,
            'scores': [],
            'D_losses': [],
            'G_losses': [],
            'real_l': [],
            'fake_l1': [],
            'fake_l2': []
        }
        ckpt = {
<<<<<<< HEAD
            'best': pack_state_dicts(netG, netD, optG, optD, schG, schD),
            'latest': pack_state_dicts(netG, netD, optG, optD, schG, schD),
=======
            'best': pack_state_dicts(netG, netD, optG, optD),
            'latest': pack_state_dicts(netG, netD, optG, optD),
>>>>>>> ccfb4c8982e1c26908e84288185ac7131ac240aa
            'history': history
        }
    h = history


    print("Starting Training Loop...")

    for ep_cnt in range(h['ep_cnt']+1 , MAX_ITER):
        h['ep_cnt'] = ep_cnt
<<<<<<< HEAD

=======
        print(f'training epoch {ep_cnt:5>}')
>>>>>>> ccfb4c8982e1c26908e84288185ac7131ac240aa
        # train for one epoch
        G_loss, D_loss, real_l, fake_l1, fake_l2 = train(dl_train,
                                                         netG,
                                                         netD,
                                                         optG,
                                                         optD,
<<<<<<< HEAD
                                                         schG,
                                                         schD
=======
>>>>>>> ccfb4c8982e1c26908e84288185ac7131ac240aa
                                                         )
        h['G_losses'].append(D_loss)
        h['D_losses'].append(G_loss)
        h['real_l'].append(real_l)
        h['fake_l1'].append(fake_l1)
        h['fake_l2'].append(fake_l2)

        plot_grid(hp, h, dl_test, netG)
        # evaluation
        score = eval(dl_test, netG, ev)

        h['scores'].append(score)


        if score > h['best_score']:
            h['best_score'] = score
            ckpt['best'] = pack_state_dicts(netG,
                                            netD,
                                            optG,
<<<<<<< HEAD
                                            optD,
                                            schG,
                                            schD
                                            )

        print(f'ep{ep_cnt:3>}, score={score:4.2f}, best={h["best_score"]:4.2f}, real_l={real_l:4.2f}, fake_l1={fake_l1:4.2f}, fake_l2={fake_l2:4.2f}')
        ckpt['latest'] = pack_state_dicts(netG,
                                          netD,
                                          optG,
                                          optD,
                                          schG,
                                          schD
=======
                                            optD
                                            )

        print(f'score={score:6.2f}, best={h["best_score"]:6.2f}, real_l={real_l:6.2f}, fake_l1={fake_l1:6.2f}, fake_l2={fake_l2:6.2f}')
        ckpt['latest'] = pack_state_dicts(netG,
                                          netD,
                                          optG,
                                          optD
>>>>>>> ccfb4c8982e1c26908e84288185ac7131ac240aa
                                          )

        # save checkpoints:
        save_checkpoint(hp, ckpt)
<<<<<<< HEAD


        # plot history
        plot_losses(hp, h)
=======
    # plot history
    plot_losses(hp, h)
>>>>>>> ccfb4c8982e1c26908e84288185ac7131ac240aa

    # plot generated images
    load_checkpoint(hp, load_type='best', netG=netG)
    netG.load_state_dict(ckpt['best']['netG'])
    plot_grid(hp, h, dl_test,  netG, name='best')


criterion = torch.nn.BCELoss()

<<<<<<< HEAD
def train(dl_train, netG, netD, optG, optD, schG=None, schD=None):
=======
def train(dl_train, netG, netD, optG, optD):
>>>>>>> ccfb4c8982e1c26908e84288185ac7131ac240aa

    real_label = 0.9
    fake_label = 0

    netG.train()
    netD.train()

    G_losses = []
    D_losses = []
    real_labels = []
    fake_labels1 = []
    fake_labels2 = []

    for real_img, c in progressBar(dl_train, len(dl_train), prefix='training', decimals=2):
        real_img = real_img.to(device)

        c = c.to(device)

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        b_size = c.shape[0]
        label = torch.full((b_size,), real_label,
                           dtype=torch.float, device=device)
        # Forward pass real batch through D


        output = netD(real_img, c).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, netG.nz, 1, 1, device=device)
        # Generate fake image batch with G
        fake = netG(noise, c.view(b_size, 24, 1, 1))
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD(fake.detach(), c).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        D_G_z1 = output.mean().item()

        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        # Update D
        optD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake, c).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()

        # Update G
        optG.step()

        # # Output training stats
        # if i % 50 == 0:
        #     print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
        #           % (epoch, num_epochs, i, len(dataloader),
        #              errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later

        G_losses.append(errG.item())
        D_losses.append(errD.item())
        real_labels.append(D_x)
        fake_labels1.append(D_G_z1)
        fake_labels2.append(D_G_z2)
        # # Check how the generator is doing by saving G's output on fixed_noise
        # if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
        #     with torch.no_grad():
        #         fake = netG(fixed_noise).detach().cpu()
        #     img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        # iters += 1
<<<<<<< HEAD
    if schG and schD:
        schG.step()
        schD.step()
=======
>>>>>>> ccfb4c8982e1c26908e84288185ac7131ac240aa
    return (np.mean(G_losses),
            np.mean(D_losses),
            np.mean(real_labels),
            np.mean(fake_labels1),
            np.mean(fake_labels2)
            )

def eval(dl_test, netG, ev):
    scores = []
    netG.eval()
    for c in progressBar(dl_test, len(dl_test), prefix='testing', decimals=1):
        c = c.to(device)
        with torch.no_grad():
            b_size = c.shape[0]
            noise = torch.randn(b_size, netG.nz, 1, 1, device=device)
            imgs = netG(noise, c.view(b_size, 24, 1, 1))
            acc = ev.eval(imgs, c)
            scores.append(acc)
    return np.mean(scores)

def plot_losses(hp,h):
    T = h['ep_cnt']+1
    plt.figure()
    plt.plot(np.arange(T), h['real_l'], 'g', label='real_l')
    plt.plot(np.arange(T), h['fake_l1'], 'y', label='fake_l1')
    plt.plot(np.arange(T), h['fake_l2'], 'r', label='fake_l2')
    plt.plot(np.arange(T), h['scores'], 'b', label='scores')
    plt.grid()
    plt.legend()
    plt.savefig(f'results/{hp_2_str(hp)}_history.png')

def plot_grid(hp, h, testdl, netG, name=''):
    imgs = []
    netG.eval()
    with torch.no_grad():
        for l in testdl:
            l = l.to(device)
            b_size = l.shape[0]
            noise = torch.randn(b_size, netG.nz, 1, 1, device=device)
            gens = netG(noise, l.view(b_size, -1, 1, 1))
            for g in gens:
                imgs.append(unnormalize(g))
        grid_img = make_grid(imgs)
        save_image(grid_img, f'results/{name}_{hp_2_str(hp)}_ep={h["ep_cnt"]}_grid.png')


if __name__ == '__main__':

    hp = {
        'lr': 0.002,
        'beta1': 0.5,
        'name': 'initial',
        'nz': 100,  # random vector size
        'ngf': 64,  # generator features
        'ndf': 64   # discriminator features
    }

<<<<<<< HEAD
    # train_with_hp(hp)
=======
    train_with_hp(hp)
>>>>>>> ccfb4c8982e1c26908e84288185ac7131ac240aa

    hp = {
        'lr': 0.002,
        'beta1': 0.5,
        'name': 'smooth+crop+smallLR',
        'nz': 100,  # random vector size
        'ngf': 64,  # generator features
        'ndf': 64   # discriminator features
    }

<<<<<<< HEAD
    # train_with_hp(hp)
=======
    train_with_hp(hp)
>>>>>>> ccfb4c8982e1c26908e84288185ac7131ac240aa

    hp = {
        'lr': 0.002,
        'beta1': 0.5,
        'name': 'smooth+smallLR',
        'nz': 100,  # random vector size
        'ngf': 64,  # generator features
        'ndf': 64   # discriminator features
    }

<<<<<<< HEAD
    # train_with_hp(hp)
=======
    train_with_hp(hp)
>>>>>>> ccfb4c8982e1c26908e84288185ac7131ac240aa

    hp = {
        'lr': 0.002,
        'beta1': 0.5,
        'name': 'smooth+quaterLR',
        'nz': 100,  # random vector size
        'ngf': 64,  # generator features
        'ndf': 64   # discriminator features
    }
<<<<<<< HEAD
    # train_with_hp(hp)

    hp = {
        'lr': 0.002,
        'beta1': 0.5,
        'name': 'smooth+octalLR',
        'nz': 100,  # random vector size
        'ngf': 64,  # generator features
        'ndf': 64   # discriminator features
    }
    #train_with_hp(hp)

    hp = {
        'lr': 0.002,
        'beta1': 0.5,
        'name': 'smooth+LR=1over16',
        'nz': 100,  # random vector size
        'ngf': 64,  # generator features
        'ndf': 64   # discriminator features
    }
    #train_with_hp(hp)


    hp = {
        'lr': 0.001,
        'beta1': 0.5,
        'name': 'smooth+LR=1over8_tinyLR',
        'nz': 100,  # random vector size
        'ngf': 64,  # generator features
        'ndf': 64   # discriminator features
    }
    # train_with_hp(hp)


    hp = {
        'lr': 0.002,
        'beta1': 0.5,
        'name': 'Scheduler1',
        'nz': 100,  # random vector size
        'ngf': 64,  # generator features
        'ndf': 64,   # discriminator features
        'sch': True
    }
    #train_with_hp(hp)

    hp = {
        'lr': 0.001,
        'beta1': 0.5,
        'name': 'Scheduler2',
        'nz': 100,  # random vector size
        'ngf': 64,  # generator features
        'ndf': 64,   # discriminator features
        'sch': False
    }
    #train_with_hp(hp)

    hp = {
        'lr': 0.002,
        'beta1': 0.5,
        'name': '1over32',
        'nz': 100,  # random vector size
        'ngf': 64,  # generator features
        'ndf': 64,   # discriminator features
        'sch': False
    }
    #train_with_hp(hp)

    hp = {
        'lr': 0.002,
        'beta1': 0.5,
        'name': 'projection_1over16',
        'nz': 100,  # random vector size
        'ngf': 64,  # generator features
        'ndf': 64   # discriminator features
    }
    #train_with_hp(hp)

    hp = {
        'lr': 0.001,
        'beta1': 0.0,
        'name': 'projection_1over5_noBeta',
        'nz': 100,  # random vector size
        'ngf': 64,  # generator features
        'ndf': 64   # discriminator features
    }
    #train_with_hp(hp)

    hp = {
        'lr': 0.001,
        'beta1': 0.0,
        'name': 'projection_1over5_noBeta_small_z',
        'nz': 50,  # random vector size
        'ngf': 64,  # generator features
        'ndf': 64   # discriminator features
    }
    #train_with_hp(hp)

    hp = {
        'lr': 0.001,
        'beta1': 0.0,
        'name': 'projection_scheduler_smaller_z',
        'nz': 25,  # random vector size
        'ngf': 64,  # generator features
        'ndf': 64,   # discriminator features
        'sch': True
    }
    #train_with_hp(hp)

    hp = {
        'lr': 0.001,
        'beta1': 0.0,
        'name': 'projection_scheduler_final',
        'nz': 50,  # random vector size
        'ngf': 64,  # generator features
        'ndf': 64,   # discriminator features
        'sch': True
    }
    # train_with_hp(hp)


    hp = {
        'lr': 0.0005,
        'beta1': 0.0,
        'name': 'projection_scheduler_1over10',
        'nz': 50,  # random vector size
        'ngf': 64,  # generator features
        'ndf': 64,   # discriminator features
        'sch': True
    }
    #train_with_hp(hp)

    hp = {
        'lr': 0.001,
        'beta1': 0.5,
        'name': 'new_test_data',
        'nz': 25,  # random vector size
        'ngf': 64,  # generator features
        'ndf': 64,   # discriminator features
        'sch': True
    }
    # train_with_hp(hp)


    hp = {
        'lr': 0.002,
        'beta1': 0.5,
        'name': 'new_test_data2',
        'nz': 100,  # random vector size
        'ngf':64,  # generator features
        'ndf': 64,   # discriminator features
        'sch': True
    }
    #train_with_hp(hp)

    hp = {
        'lr': 0.001,
        'beta1': 0.5,
        'name': 'new_test_data3',
        'nz': 50,  # random vector size
        'ngf': 32,  # generator features
        'ndf': 64,   # discriminator features
        'sch': True
    }
=======
>>>>>>> ccfb4c8982e1c26908e84288185ac7131ac240aa
    train_with_hp(hp)
