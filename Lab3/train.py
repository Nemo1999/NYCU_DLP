from dataloader import get_loader
from model import initialize_resnet
from torch.utils import data  as data
import torch
import torch.nn as nn
from os.path import isfile
from itertools import islice
import sys
from util import progressBar

# hyper_parameters

hp = {
    'model':'ResNet18', # or  'ResNet50'
    'optimizer':'sgd',
    'lr':1e-3,
    'pretrain':True,
    'feature_extracting':False,
    'bs':32,
    'epochs':3
}

def hp_2_str(hp):
    return f"{hp['model']}_opt={hp['optimizer']}_lr={hp['lr']}_pretrain={hp['pretrain']}_feature_extracting={hp['feature_extracting']}_epoch={hp['epochs']}"

def get_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))
    return device
def get_loss(hp):
    return nn.CrossEntropyLoss()

def get_optimizer(hp,model):
    if hp['optimizer'] == 'sgd':
        return torch.optim.SGD(model.parameters(),lr=hp['lr'],momentum=0.9,weight_decay=5e-4)
    elif hp['optimizer'] == 'adam':
        return torch.optim.Adam(model.parameters(),lr=hp['lr'])
def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    # turn model into training mode
    model.train()
    epoch_loss = 0.0
    correct = 0.0
    for X,y in progressBar(dataloader,prefix='training',decimals=2):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += float(loss)
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        """
        if batch %10 == 0:
            loss, current = loss.item(), batch*len(X)
            print(f" train loss: {loss:>7f} \t\t [{current:>5d}/{size:>5d}]")
        """
    epoch_loss /= num_batches
    correct/=size
    #print(f"epoch loss: {epoch_loss:>7f}")
    return correct, epoch_loss


# %%
def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    batch_size = dataloader.batch_size

    # turn model into eval mode
    model.eval()
    
    test_loss, correct = 0,0
    with torch.no_grad():
        for X, y in progressBar(dataloader,prefix='testing',decimals=2):
            X, y = X.to(device), y.to(device)
            #print(y.dtype)
            pred = model(X)
            #print(pred.dtype)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            
            sys.stdout.flush()
    test_loss /= num_batches
    correct /= size
    #print(f'Test Accuracy: {(100 * correct):>0.1f}%, test loss:{test_loss:>8f} \n')
    return correct, test_loss

def save_checkpoint(epoch,hp, model, optimizer, history):
    name = hp_2_str(hp)
    path = 'checkpoints/'+hp_2_str(hp)+f'trained_ep{epoch}.pth'
    torch.save({
        'epoch':epoch,
        'model_state_dict':model.state_dict(),
        'optimizer_state_dict':optimizer.state_dict(),
        'hp':hp,
        'history': history
    },path)
    print(f'checkpoint saved at {path}')

def search_checkpoint(epoch,hp):
    name = hp_2_str(hp)
    path = 'checkpoints/'+hp_2_str(hp)+f'trained_ep{epoch}.pth'
    return isfile(path)

def load_checkpoint(epoch,hp,model=None,optimizer=None):
    name = hp_2_str(hp)
    path = 'checkpoints/'+hp_2_str(hp)+f'trained_ep{epoch}.pth'
    ckpt = torch.load(path)
    if model:
        model.load_state_dict(ckpt['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    return  model, optimizer, ckpt['history']

def train_with_hp(hp,save_result=True):
   
    # get dataloader
    train_dl, test_dl = get_loader(hp)
    # get device
    device = get_device()
    # generate model
    model = initialize_resnet(hp).to(device)
    print(model)
    # loss function
    loss_fn = get_loss(hp)
    # optimizer
    optimizer = get_optimizer(hp,model)
    history = {
        'test_acc':[],
        'test_loss':[],
        'train_acc':[],
        'train_loss':[]
    }

    epochs = hp['epochs']
    for t in range(epochs):
        if  search_checkpoint(t,hp):
            if not search_checkpoint(t+1,hp):
                model, optimizer, history = load_checkpoint(t,hp,model,optimizer)
                print('previous history: -----------------')
                for k in history:
                    print(f'{k}: {history[k]}')
            continue
        
        print(f"Epoch {t}------------------------------------ ")
        train_acc, train_loss = train(train_dl, model, loss_fn, optimizer,device)
        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        print(f"train_acc: {train_acc:<5.4f}")        

        test_acc, test_loss = test(test_dl, model, loss_fn, device)
        history['test_acc'].append(test_acc)
        history['test_loss'].append(test_loss)

        print(f"test_acc: {test_acc:<5.4f}")
        #print('\n',end='')

        if save_result:
            save_checkpoint(t,hp,model,optimizer,history)
        print('\n')
    return model, history



if __name__ == '__main__':
    
    hp = {
        'model':'ResNet18', # or  'ResNet50'
        'optimizer':'sgd',
        'lr':1e-3,
        'pretrain':True,
        'feature_extracting':False,
        'bs':4,
        'epochs':10
    }
    model, hist = train_with_hp(hp,save_result=True)
    
    hp = {
        'model':'ResNet18', # or  'ResNet50'
        'optimizer':'sgd',
        'lr':1e-3,
        'pretrain':False,
        'feature_extracting':False,
        'bs':4,
        'epochs':10
    }

    model, hist = train_with_hp(hp,save_result=True)
    
    
    hp = {
        'model':'ResNet50', # or  'ResNet50'
        'optimizer':'sgd',
        'lr':1e-3,
        'pretrain':True,
        'feature_extracting':False,
        'bs':4,
        'epochs':10
    }

    model, hist = train_with_hp(hp,save_result=True)

    hp = {
        'model':'ResNet50', # or  'ResNet50'
        'optimizer':'sgd',
        'lr':1e-3,
        'pretrain':False,
        'feature_extracting':False,
        'bs':8,
        'epochs':5
    }
    model, hist = train_with_hp(hp,save_result=True)
    
    #---------------------------------------------------

    #feature_extraction
    hp = {
        'model':'ResNet18', # or  'ResNet50'
        'optimizer':'sgd',
        'lr':1e-3,
        'pretrain':True,
        'feature_extracting':True,
        'bs':4,
        'epochs':10
    }
    model, hist = train_with_hp(hp,save_result=True)
    
    hp = {
        'model':'ResNet50', # or  'ResNet50'
        'optimizer':'sgd',
        'lr':1e-3,
        'pretrain':True,
        'feature_extracting':True,
        'bs':4,
        'epochs':5
    }

    model, hist = train_with_hp(hp,save_result=True)

