import torch
from get_device import device
from dataloader import get_training_pairs, get_testing_pairs, nums2word
from torch.utils import data
from model import EncoderRNN, DecoderRNN
from sample import compute_bleu, Gaussian_score
from progressbar import progressBar
from itertools import islice
from os.path import isfile
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import random

hp = {
    'kld_w': [0.1, 0.05, 0.4, 20], # start, step, end, turning epoch
    'kld_pat': 'monotonic',  # 'cyclical'
    'teacher_w': [1.0, -0.05, 0.5, 100],
    'lr':  0.01,
    'latent_size': 128,

}

MAX_ITER = 200

def hp_2_str(hp):
    k = hp['kld_w']
    kp = hp['kld_pat']
    tw = hp['teacher_w']
    lr = hp['lr']
    ls = hp['latent_size']
    name = hp['name']
    return f'k={k}_kp={kp}_tw={tw}_lr={lr}_ls={ls}_name={name}'

def search_checkpoint(hp):
    name = hp_2_str(hp)
    path = 'checkpoints/'+hp_2_str(hp)+'_ckpt.pth'
    return isfile(path)

def save_checkpoint(hp, ckpt):
    name = hp_2_str(hp)
    path = 'checkpoints/'+hp_2_str(hp)+'_ckpt.pth'
    torch.save(ckpt, path)
    #print(f'checkpoint is save at {path}')


def load_checkpoint(hp, load_type='latest', encoder=None, decoder=None, opt_enc=None, opt_dec=None):
    name = hp_2_str(hp)
    path = 'checkpoints/'+hp_2_str(hp)+'_ckpt.pth'
    ckpt = torch.load(path)
    if encoder:
        encoder.load_state_dict(ckpt[load_type]['enc'])
    if decoder:
        decoder.load_state_dict(ckpt[load_type]['dec'])
    if opt_enc:
        opt_enc.load_state_dict(ckpt[load_type]['opt_enc'])
    if opt_dec:
        opt_dec.load_state_dict(ckpt[load_type]['opt_dec'])
    return ckpt


def get_next_kld_w(pattern, ep_cnt, params):
    if pattern == 'cyclical':
        start = params[0]
        step = params[1]
        stop = params[2]
        duration = params[3]
        if (ep_cnt % duration) < ((stop - start) / step):
            return start + step * (ep_cnt % duration)
        else:
            return stop
    else:
        assert pattern == 'monotonic', 'unknown pattern for kld_weight '
        start = params[0]
        step = params[1]
        stop = params[2]
        ep_start = params[3]
        if ep_cnt < ep_start:
            return start
        elif ep_cnt < ep_start + (stop - start) / step:
            return start + step * (ep_cnt - ep_start)
        else:
            return stop

def get_next_teacher_w(ep_cnt, params):
    return get_next_kld_w('monotonic', ep_cnt, params)

def pack_state_dicts(enc, dec, opt_enc, opt_dec):
    state = {
        'enc': enc.state_dict(),
        'dec': dec.state_dict(),
        'opt_enc': opt_enc.state_dict(),
        'opt_dec': opt_dec.state_dict()
    }
    return deepcopy(state)

def train_with_hp(hp):
    dl_train = data.DataLoader(get_training_pairs(),batch_size=1, shuffle=True)
    dl_test = data.DataLoader(get_testing_pairs(),batch_size=1, shuffle = True)

    enc = EncoderRNN(hp['latent_size']).to(device)
    dec = DecoderRNN(hp['latent_size']).to(device)

    opt_enc = torch.optim.SGD(enc.parameters(), lr=hp['lr'])
    opt_dec = torch.optim.SGD(dec.parameters(), lr=hp['lr'])

    if search_checkpoint(hp):
        ckpt = load_checkpoint(hp, 'latest', enc, dec, opt_enc, opt_dec)
        history = ckpt['history']
    else:
        history = {
             'ep_cnt': -1,
            'teacher_w': [],
            'kld_w': [],
            'bleu_score': [],
            'best_bleu': -float('inf'),
            'gaussian_score': [],
            'best_gaussian': -float('inf'),
            'reg_loss': [],
            'rec_loss': []
        }
        ckpt = {
            'best_blue': pack_state_dicts(enc, dec, opt_enc, opt_dec),
            'best_gaussian': pack_state_dicts(enc, dec, opt_enc, opt_dec),
            'latest': pack_state_dicts(enc, dec, opt_enc, opt_dec),
            'history': history
        }


    h = history

    for ep_cnt in range(h['ep_cnt']+1,MAX_ITER):

        h['ep_cnt'] = ep_cnt
        # find next kld-weight  and teacher-forcing-ratio
        kld_next = get_next_kld_w(hp['kld_pat'], ep_cnt , hp['kld_w'])
        h['kld_w'].append(kld_next)
        teacher_next = get_next_teacher_w(ep_cnt, hp['teacher_w'])
        h['teacher_w'].append(teacher_next)

        # train for one epoch
        res = train(dl_train,
                    enc,
                    dec,
                    opt_enc,
                    opt_dec,
                    kld_next,
                    teacher_next,

                    max_samples=None)
        reg_loss, rec_loss, total_loss = res
        h['reg_loss'].append(reg_loss.item())
        h['rec_loss'].append(rec_loss.item())

        # evaluation
        res = test(dl_test, enc, dec)
        bleu_score, gaussian_score = res
        h['bleu_score'].append(bleu_score)
        h['gaussian_score'].append(gaussian_score)

        if bleu_score > h['best_bleu']:
            h['best_bleu'] = bleu_score
            ckpt['best_bleu'] = pack_state_dicts(enc,
                                                 dec,
                                                 opt_enc,
                                                 opt_dec)
        if gaussian_score > h['best_gaussian']:
            h['best_gaussian'] = gaussian_score
            ckpt['best_gaussian'] = pack_state_dicts(enc,
                                                     dec,
                                                     opt_enc,
                                                     opt_dec)
        ckpt['latest'] = pack_state_dicts(enc,
                                          dec,
                                          opt_enc,
                                          opt_dec)
        print(f'ep{ep_cnt:>3}, gaussian={gaussian_score:>4.2f}, bleu_score={bleu_score:>4.2f}, reg_loss={reg_loss:>4.2f}, rec_loss={rec_loss:>4.2f}')

        # save checkpoints:
        save_checkpoint(hp, ckpt)
        plot_losses(hp, h)
    plot_losses(hp, h)

def train(dl_train,
          enc,
          dec,
          opt_enc,
          opt_dec,
          kld_w,
          teacher_w,
          max_samples=None):

    # set models to training mode
    enc.train()
    dec.train()


    if max_samples is not None:
        dl_train = islice(dl_train, 0, max_samples)
        size = max_samples
    else:
        size = len(dl_train)
    kl_losses = 0.0
    rc_losses = 0.0
    total_losses = 0.0
    for w, t in progressBar(dl_train, size, prefix='training', decimals=2):
        #w_str = nums2word(w.view(-1))[:-1]
        #t_str =['sp','tp','pg','p'][t.item()]
        #print(w_str, t_str)
        w = w.transpose(0, 1).to(device)
        t = t.to(device)
        h0 = enc.initHidden()
        c0 = enc.initCell()
        kl_loss, zh, zc = enc('train', w, h0, c0, t)

        use_teacher_forcing = True if random.random() < teacher_w else False

        #h0 = dec.initHidden()  # init hidden code by zeros
        h0 = zh.view(1, 1, -1) # init hidden code by latent z
        c0 = zc.view(1, 1, -1)
        rc_loss = dec('train', h0, c0, t, w, use_teacher_forcing)

        loss = kld_w * kl_loss + rc_loss

        opt_enc.zero_grad()
        opt_dec.zero_grad()
        loss.backward()
        opt_enc.step()
        opt_dec.step()

        total_losses += loss
        kl_losses += kl_loss
        rc_losses += rc_loss

    total_losses /= size
    kl_losses /= size
    rc_losses /= size

    return kl_losses, rc_losses, total_losses

def test(dl_test, enc, dec):
    size = len(dl_test)


    # set models to eval mode

    enc.eval()
    dec.eval()

    bleu_score = 0
    with torch.no_grad():
        #  compute bleu score
        for cnt, ((w1, t1), (w2, t2)) in enumerate(dl_test):
            reference = nums2word(w2.view(-1))[:-1]
            w1 = w1.transpose(0, 1).to(device)
            t1 = t1.to(device)
            t2 = t2.to(device)

            h0 = enc.initHidden()
            c0 = enc.initCell()
            zh, zc = enc('eval', w1, h0, c0, t1)

            # h0 = dec.initHidden()
            h0 = zh.view(1, 1, -1)
            c0 = zc.view(1, 1, -1)
            output = dec('eval', h0, c0, t2)
            output = nums2word(output)
            bleu_score += compute_bleu(output, reference)
            if cnt < 10:
                print('bleu_test: ',output, ' / ', reference )
        bleu_score /= size

        # compute gaussian score
        words = []
        for i in range(100):
            # generate gaussian noise in latent space
            zh = torch.randn(1, 1, enc.latent_size)
            zh = zh.to(device)

            zc = torch.randn(1, 1, enc.latent_size)
            zc = zc.to(device)

            words.append([])
            for j in range(4):
                t = torch.tensor([[[j]]], device=device)
                # h0 = dec.initHidden()
                h0 = zh
                c0 = zc
                output = dec('eval', h0, c0, t)
                output = nums2word(output)
                words[-1].append(output)
        gaussian_score = Gaussian_score(words)
        print('gaussian test: ', words[-5:-1])
    return bleu_score, gaussian_score

def plot_losses(hp, h):
    T = h['ep_cnt']+1
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.plot(np.arange(T), h['rec_loss'], 'green', label='rec_loss')
    ax.plot(np.arange(T), h['reg_loss'], 'blue', label='reg_loss')
    ax.set_ylabel("training losses",color="blue",fontsize=14)

    ax2 = ax.twinx()
    ax2.set_ylabel("training weights and scores",color="red",fontsize=14)

    ax2.plot(np.arange(T), h['kld_w'], 'r-', label='kld_w', linewidth=1)
    ax2.plot(np.arange(T), h['teacher_w'], 'r', label='teacher_w',linewidth=1)

    ax2.plot(np.arange(T), h['gaussian_score'], 'orange', label='gaussian_score')
    ax2.plot(np.arange(T), h['bleu_score'], 'yellow', label='bleu_score')

    ax.grid()
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    # plt.legend()
    plt.savefig(f'results/{hp_2_str(hp)}_history.png')
    plt.close()

if __name__ == '__main__':





    hp = {
        'kld_w': [0.02, 0.005, 0.4, 20], # start, step, end, turning epoch
        'kld_pat': 'monotonic',  # 'cyclical'
        'teacher_w': [1.0, -0.02, 0.5, 100],
        'lr':  0.01,
        'latent_size': 64,
        'name': 'v2'
    }

    #train_with_hp(hp)

    hp = {
        'kld_w': [0.0, 0.02, 0.4, 20], # start, step, end, turning epoch
        'kld_pat': 'cyclical',  # 'cyclical'
        'teacher_w': [1.0, -0.02, 0.5, 100],
        'lr':  0.01,
        'latent_size': 64,
        'name': 'v2'
    }
# no_sample-------------------------------------------------------------
    #train_with_hp(hp)
    hp = {
        'kld_w': [0.02, 0.005, 0.4, 20], # start, step, end, turning epoch
        'kld_pat': 'monotonic',  # 'cyclical'
        'teacher_w': [1.0, -0.02, 0.5, 100],
        'lr':  0.01,
        'latent_size': 32,
        'name': 'no_sample32'
    }


    #train_with_hp(hp)

    hp = {
        'kld_w': [0.02, 0.005, 0.4, 20], # start, step, end, turning epoch
        'kld_pat': 'monotonic',  # 'cyclical'
        'teacher_w': [1.0, -0.02, 0.5, 100],
        'lr':  0.01,
        'latent_size': 64,
        'name': 'no_sample64'
    }
    #train_with_hp(hp)

    hp = {
        'kld_w': [0.02, 0.005, 0.4, 20], # start, step, end, turning epoch
        'kld_pat': 'monotonic',  # 'cyclical'
        'teacher_w': [1.0, -0.02, 0.5, 100],
        'lr':  0.01,
        'latent_size': 128,
        'name': 'no_sample128'
    }
    #train_with_hp(hp)
# slowerSG-----------------------------------------------------------


    hp = {
        'kld_w': [0.02, 0.005, 0.4, 100], # start, step, end, turning epoch
        'kld_pat': 'monotonic',  # 'cyclical'
        'teacher_w': [1.0, -0.02, 0.5, 100],
        'lr':  0.01,
        'latent_size': 128,
        'name': 'sample128_slowerRG'
    }

    #train_with_hp(hp)

    hp = {
        'kld_w': [0.02, 0.005, 0.4, 100], # start, step, end, turning epoch
        'kld_pat': 'monotonic',  # 'cyclical'
        'teacher_w': [1.0, -0.02, 0.5, 100],
        'lr':  0.03,
        'latent_size': 32,
        'name': 'sample32_slowerRG_largeLR'
    }

    train_with_hp(hp)

    hp = {
        'kld_w': [0.02, 0.005, 0.4, 100], # start, step, end, turning epoch
        'kld_pat': 'monotonic',  # 'cyclical'
        'teacher_w': [1.0, -0.02, 0.5, 100],
        'lr':  0.07,
        'latent_size': 64,
        'name': 'sample128_slowerRG_bigLR'
    }
    #train_with_hp(hp)
