import numpy as np
import torch.nn as nn
import torch
from dataloader import get_loader
from train import get_device, get_optimizer, search_checkpoint, load_checkpoint
from model import initialize_resnet
import matplotlib.pyplot as plt 


def plot_learning_curves(hp):
    history={
        'pretrain':None,
        'no-pretrain':None
    }
    
    pretrain = ['pretrain', 'no-pretrain']
    for p, tf in zip(pretrain,[True,False]):
        hp['pretrain']=tf
        assert search_checkpoint(hp['epochs']-1,hp), "training process not finished"
        _, _, history[p] = load_checkpoint(hp['epochs']-1,hp)
    
        
    plt.figure(figsize=(10,8))
    
    
    for p_key in pretrain:
        style = 'r' if p_key == 'pretrain' else 'b'
        for t in ['test','train']:
            t_key = t + '_acc'
            style_ = style +'-.' if t == 'test' else style
            plt.plot(
                history[p_key][t_key],
                style_,
                label=f'{t}-{p_key}',
                linewidth=1
            )
            print(f'Highest {t}ing ACC for {hp["model"]} using {p_key:<16}: {max(history[p_key][t_key]):>7.3f}')
    plt.xlabel('epochs')
    plt.ylabel('acc')
    plt.yticks(np.arange(0.7,0.85,0.02))
    plt.title(f'Result Comparason ({hp["model"]})',fontsize=20)
    plt.legend()
    plt.grid()
    plt.savefig(f'plots/Compare_{hp["model"]}.png',backend='agg')
if __name__ == '__main__':
    #ResNet18
    hp = {
        'model':'ResNet18', # or  'ResNet50'
        'optimizer':'sgd',
        'lr':1e-3,
        'pretrain':True,
        'feature_extracting':False,
        'bs':128,
        'epochs':10
    }

    plot_learning_curves(hp)

    # ResNet50
    
    hp = {
        'model':'ResNet50', # or  'ResNet50'
        'optimizer':'sgd',
        'lr':1e-3,
        'pretrain':True,
        'feature_extracting':False,
        'bs':4,
        'epochs':5
    }

    plot_learning_curves(hp)
