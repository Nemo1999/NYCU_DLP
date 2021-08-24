from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch.nn as nn
import torch
import matplotlib.pyplot as plt 
from dataloader import get_loader
from train import get_device, search_checkpoint, load_checkpoint
from model import initialize_resnet
from util import progressBar
import numpy as np
def plot_confusion_matrix(hp,t=None):
    print('\n\n')
    print(f"Trying to test {hp['model']}_pretrain={hp['pretrain']}:")
    print('--------------------------------------------------------')
    # get dataloader
    train_dl, test_dl = get_loader(hp)
    # get device
    device = get_device()
    # generate model
    model = initialize_resnet(hp).to(device)
    #print(model)
    
    history = {
        'test_acc':[],
        'test_loss':[],
        'train_acc':[],
        'train_loss':[]
    }
    
    # load model parameters 
    if t:
        epochs = t
    else :
        epochs = hp['epochs']
    
    for t in range(epochs):
        if  search_checkpoint(t,hp):
            if not search_checkpoint(t+1,hp):
                model, _ , history = load_checkpoint(t,hp,model)
                print('previous test acc : -----------------')
                for cnt,  l in enumerate(history['test_acc']):
                    print(f'epoch{cnt:<5}:{l:<5.3f}')
                    
            continue

    model.eval()
    # generate predict labels
    y_pred = []
    y_truth = []
    with torch.no_grad():
        for X, y in progressBar(test_dl,prefix='testing',decimals=2):
            X, y = X.to(device), y.to(device)
            #print(y.dtype)
            pred = model(X)
            #print(pred.dtype)
            y_pred.extend(list(pred.argmax(1).detach().cpu().numpy()))
            y_truth.extend(list(y.detach().cpu().numpy()))
    # get ground truth label
    
    cf = confusion_matrix(y_truth,y_pred, normalize='true')
    disp = ConfusionMatrixDisplay(cf,display_labels=np.arange(5))
    disp.plot(cmap=plt.cm.Blues)
    disp.ax_.set_title(f"Normalized Confusion Matrix {hp['model']} pretrain={hp['pretrain']}")
    plt.savefig(f"plots/CM_{hp['model']}_pretrain={hp['pretrain']}.png",backend='agg')
    acc = np.sum(np.array(y_truth) == np.array(y_pred)) / len(y_truth)
    return acc 
    
if __name__ == '__main__':

    # ResNet18
    hp = {
        'model':'ResNet18', # or  'ResNet50'
        'optimizer':'sgd',
        'lr':1e-3,
        'pretrain':True,
        'feature_extracting':False,
        'bs':128,
        'epochs':10
    }
    
    acc = plot_confusion_matrix(hp)
    print(f"accuracy for {hp['model']}_pretrain={hp['pretrain']} = {acc}")

"""
    hp = {
        'model':'ResNet18', # or  'ResNet50'
        'optimizer':'sgd',
        'lr':1e-3,
        'pretrain':False,
        'feature_extracting':False,
        'bs':128,
        'epochs':10
    }
    
    acc = plot_confusion_matrix(hp)
    print(f"accuracy for {hp['model']}_pretrain={hp['pretrain']} = {acc}")


    # ResNet 50
    hp = {
        'model':'ResNet50', # or  'ResNet50'
        'optimizer':'sgd',
        'lr':1e-3,
        'pretrain':True,
        'feature_extracting':False,
        'bs':64,
        'epochs':5
    }

    acc = plot_confusion_matrix(hp)
    print(f"accuracy for {hp['model']}_pretrain={hp['pretrain']} = {acc}")


    # ResNet 50
    hp = {
        'model':'ResNet50', # or  'ResNet50'
        'optimizer':'sgd',
        'lr':1e-3,
        'pretrain':False,
        'feature_extracting':False,
        'bs':64,
        'epochs':5
    }

    acc = plot_confusion_matrix(hp)
    print(f"accuracy for {hp['model']}_pretrain={hp['pretrain']} = {acc}")
"""
