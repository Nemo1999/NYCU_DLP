import torch
from train import load_checkpoint

def plot_history(h):



if __name__ == '__main__':
    hp = {
        'kld_w': [0.1, 0.005, 0.4, 20], # start, step, end, turning epoch
        'kld_pat': 'monotonic',  # 'cyclical'
        'teacher_w': [1.0, -0.02, 0.5, 100],
        'lr':  0.01,
        'latent_size': 128,
        'total_epochs': 150
    }

    ckpt = load_checkpoint(hp)
    print(ckpt['history']['reg_loss'])
