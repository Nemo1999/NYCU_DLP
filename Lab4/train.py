import torch
from dataloader import get_training_pairs, get_testing_pairs, nums2word
from torch.utils import data
from model import EncoderRNN, DecoderRNN
from sample import compute_blue, Gaussian_score
hp = {
    'kld_w': [0.1, 0.05, 0.4, 20], # start, step, end, turning epoch
    'kld_pat': 'monotonic',  # 'cyclical'
    'teacher_w': [1.0, -0.05, 0.5, 100],
    'lr':  0.01
    'latent_size': 128
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

def save_checkpoint(hp, best_bleu, best_gaussian, latest, history):
    name = hp_2_str(hp)
    path = 'checkpoints/'+hp_2_str(hp)+'_ckpt.pth'
    torch.save({
        'best_bleu': best_bleu,
        'best_gaussian': best_gaussian,
        'latest': latest,
        'history': history
    },path)
    print(f'checkpoint is save at {path}')


def load_checkpoint(hp, load_type='latest', encoder=None, decoder=None, opt_enc=None, opt_dec=None):
    name = hp_2_str(hp)
    path = 'checkpoints/'+hp_2_str(hp)+'_ckpt.pth'
    ckpt = torch.load(path)
    if encoder:
        encoder.load_state_dict(ckpt[load_type]['encoder'])
    if decoder:
        decoder.load_state_dict(ckpt[load_type]['decoder'])
    if opt_enc:
        opt_enc.load_state_dict(ckpt[load_type]['opt_enc'])
    if opt_dec:
        opt_dec.load_state_dict(ckpt[load_type]['opt_dec'])
    return ckpt['history']

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
        assert pattern == 'monotonic':
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

def train_with_hp(hp):
    dl_train = data.DataLoader(get_training_pairs(),batch_size=1, shuffle=True)
    dl_test = data.DataLoader(get_testing_pairs(),batch_size=1, shuffle = False)

    enc = EncoderRNN(hp['latent_size'])
    dec = DecoderRNN(hp['latent_size'])

    opt_enc = torch.optim.SGD(enc.parameters(), lr=hp['lr'])
    opt_dec = torch.optim.SGD(dec.parameters(), lr=hp['lr'])

    if search_checkpoint(hp):
        history = load_checkpoint(hp, 'latest', enc, dec, opt_enc, opt_dec)
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


    h = history

    for ep_cnt in range(h['ep_cnt']+1, hp['total_epochs']):

        h['ep_cnt'] = ep_cnt
        # find next kld-weight  and teacher-forcing-ratio
        kld_next = get_next_kld_w(hp['kld_pat'], ep_cnt , hp['kld_w'])
        h['kld_w'].append(kld_next)
        teacher_next = get_next_teacher_w(ep_cnt, hp['teacher_w'])
        h['teacher_w'].append(teacher_next)

    # train for one epoch
    res = train(dl_train, enc, dec, opt_enc, opt_dec, kld_next, teacher_next)
    reg_loss, rec_loss = res
    h['reg_loss'].append(reg_loss)
    h['rec_loss'].append(rec_loss)

    # evaluation
    res = test(dl_test, enc, dec)
    bleu_score, gaussian_score = res
    h['bleu_score'].append(bleu_score)
    h['gaussian_score'].append(gaussian_score)

    if bleu_score > h['best_bleu']:
        h['best_bleu'] = bleu_score
    if gaussian_score > h['best_gaussian']:
        h['best_gaussian'] = gaussian_score

    # save checkpoints:

def train():
    for w, t in dl_train:
        w = w.transpose(0,1).to(device)
        t = t.to(device)
        h0 = enc.initHidden()
        c0 = enc.initCell()
        kl_loss, code = enc(w, h0, c0, t)
        print(kl_loss)
        h0 = dec.initHidden()
        c0 = code.view(1,1,-1)
        rc_loss = dec(h0, c0, t, w)
        print(rc_loss)
        break

if __name__ == '__main__':
    for w,t in dl_train:
        w = w.transpose(0,1)
        print(w.shape)
        print(t.shape)
        break
