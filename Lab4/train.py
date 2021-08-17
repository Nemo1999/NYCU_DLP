import torch
from get_device import device
from dataloader import get_training_pairs, get_testing_pairs, nums2word
from torch.utils import data
from model import EncoderRNN, DecoderRNN
from sample import compute_bleu, Gaussian_score
from progressbar import progressBar
from itertools import islice
from os.path import isfile
import random

hp = {
    'kld_w': [0.1, 0.05, 0.4, 20], # start, step, end, turning epoch
    'kld_pat': 'monotonic',  # 'cyclical'
    'teacher_w': [1.0, -0.05, 0.5, 100],
    'lr':  0.01,
    'latent_size': 128,
    'total_epochs': 150
}


def hp_2_str(hp):
    k = hp['kld_w']
    kp = hp['kld_pat']
    tw = hp['teacher_w']
    lr = hp['lr']
    ls = hp['latent_size']
    return f'k={k}_kp={kp}_tw={tw}_lr={lr}_ls={ls}'

def search_checkpoint(hp):
    name = hp_2_str(hp)
    path = 'checkpoints/'+hp_2_str(hp)+'_ckpt.pth'
    return isfile(path)

def save_checkpoint(hp, ckpt):
    name = hp_2_str(hp)
    path = 'checkpoints/'+hp_2_str(hp)+'_ckpt.pth'
    torch.save(ckpt, path)
    print(f'checkpoint is save at {path}')


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
        if ep_cnt % duration < (stop - start) / step:
            return start + step * ep_cnt
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
    return state

def train_with_hp(hp):
    dl_train = data.DataLoader(get_training_pairs(),batch_size=1, shuffle=True)
    dl_test = data.DataLoader(get_testing_pairs(),batch_size=1, shuffle = False)

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

    for ep_cnt in range(h['ep_cnt']+1, hp['total_epochs']):

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
                    max_samples=50)
        reg_loss, rec_loss, total_loss = res
        h['reg_loss'].append(reg_loss)
        h['rec_loss'].append(rec_loss)

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

        # save checkpoints:
        save_checkpoint(hp, ckpt)


def train(dl_train,
          enc,
          dec,
          opt_enc,
          opt_dec,
          kld_w,
          teacher_w,
          max_samples=None):

    if max_samples is not None:
        dl_train = islice(dl_train, 0, max_samples)
        size = max_samples
    else:
        size = len(dl_train)
    kl_losses = 0.0
    rc_losses = 0.0
    total_losses = 0.0
    for w, t in progressBar(dl_train, size, prefix='training', decimals=2):
        w = w.transpose(0, 1).to(device)
        t = t.to(device)
        h0 = enc.initHidden()
        c0 = enc.initCell()
        kl_loss, code = enc('train', w, h0, c0, t)

        use_teacher_forcing = True if random.random() < teacher_w else False

        h0 = dec.initHidden()  # init hidden code by zeros
        # h0 = code.view(1, 1, -1) # init hidden code by latent z
        c0 = code.view(1, 1, -1)
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
    kl_losses /= kl_loss
    rc_losses /= rc_loss

    return kl_losses, rc_losses, total_losses

def test(dl_test, enc, dec):
    size = len(dl_test)

    enc.eval()
    dec.eval()

    bleu_score = 0
    with torch.no_grad():
        #  compute bleu score
        for (w1, t1), (w2, t2) in dl_test:
            reference = nums2word(w2.view(-1))[:-1]
            w1 = w1.transpose(0, 1).to(device)
            t1 = t1.to(device)
            t2 = t2.to(device)

            h0 = enc.initHidden()
            c0 = enc.initCell()
            code = enc('eval', w1, h0, c0, t1)

            h0 = dec.initHidden()
            c0 = code.view(1, 1, -1)
            output = dec('eval', h0, c0, t2)
            output = nums2word(output)
            bleu_score += compute_bleu(output, reference)
        bleu_score /= size

        # compute gaussian score
        words = []
        for i in range(100):
            # generate gaussian noise in latent space
            noise = torch.randn(1, 1, enc.latent_size)
            words.append([])
            for j in range(4):
                t = torch.tensor([[[j]]], device=device)
                h0 = dec.initHidden()
                c0 = noise
                output = dec('eval', h0, c0, t)
                output = nums2word(output)
                words[-1].append(output)
        gaussian_score = Gaussian_score(words)

    return bleu_score, gaussian_score


if __name__ == '__main__':

    hp = {
        'kld_w': [0.1, 0.05, 0.4, 20], # start, step, end, turning epoch
        'kld_pat': 'monotonic',  # 'cyclical'
        'teacher_w': [1.0, -0.05, 0.5, 100],
        'lr':  0.01,
        'latent_size': 128,
        'total_epochs': 150
    }

    train_with_hp(hp)
