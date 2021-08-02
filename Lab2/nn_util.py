# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from dataloader import read_bci_data
from  torch.utils.data import DataLoader, Dataset, TensorDataset
from torch import tensor
import torch
import torch.nn as nn
import numpy as np


# %%

def hp_2_str(hp):
    return f"{hp['NN']}_{hp['act']}_lr={hp['lr']}_opt={hp['optimizer']}_ep={hp['epochs']}"


class random_shift_dataset(Dataset):
    def __init__(self,x,y):
        assert x.shape[0] == y.shape[0], 'invalid dataset'
        self.x = x
        self.y = y
    def __len__(self):
        return self.x.shape[0]
    def __getitem__(self,idx):
        
        shift = np.random.randint(low = -self.x.shape[-1]/8, high = self.x.shape[-1]/8)
        x = np.roll(self.x[idx],shift,axis=-1)
        return tensor(x,dtype=torch.float), tensor(self.y[idx],dtype=torch.long)

# %%
def gen_loader(hp):
    batch_size = hp['bs']
    #read data from file
    train_x, train_y, test_x, test_y =  read_bci_data()


    # Dataloader 
    train_dl = DataLoader(
        random_shift_dataset(train_x, train_y),
        batch_size=batch_size, shuffle= True
    )

    test_dl = DataLoader(
        TensorDataset(tensor(test_x,dtype=torch.float),
                      tensor(test_y,dtype=torch.long)
        ),
        batch_size=batch_size, shuffle = False
    )

    return train_dl, test_dl

def get_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))
    return device


# %%
def get_activation(act_name):
    if act_name == 'ELU':
        return nn.ELU()
    elif act_name == 'ReLU':
        return nn.ReLU()
    else:
        return nn.LeakyReLU()

class EEGNet(nn.Module):
    def __init__(self,hp):
        super(EEGNet, self).__init__()
        self.firstConv = nn.Sequential(
            nn.Conv2d(1,16,kernel_size=(1,51), stride=(1,1), padding=(0,25), bias=False),
            nn.BatchNorm2d(16,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(16,32, kernel_size=(2,1), stride=(1,1), groups=16, bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            get_activation(hp['act']),
            nn.AvgPool2d(kernel_size=(1,4), stride=(1,4), padding=0),
            nn.Dropout(p=0.25)
        )
        self.separableConv = nn.Sequential(
            nn.Conv2d(32,32,kernel_size=(1,15), stride=(1,1), padding=(0,7), bias=False),
            nn.BatchNorm2d(32,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            get_activation(hp['act']),
            nn.AvgPool2d(kernel_size=(1,8), stride=(1,8),padding=0),
            nn.Dropout(p=0.25),
            nn.Flatten()
        )
        self.classify = nn.Sequential(
            nn.Linear(32*23, 2, bias=False)
        )
    def forward(self,x):
        x = self.firstConv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        x = self.classify(x)

        return x


# %%
class DeepConvNet(nn.Module):
    def __init__(self,hp):
        super(DeepConvNet,self).__init__()
        self.firstConv = nn.Sequential(
            nn.Conv2d(1,25, kernel_size=(1,8),padding='same'),
            nn.Conv2d(25,25,kernel_size=(2,1)),
            nn.BatchNorm2d(25, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            get_activation(hp['act']),
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(p=0.3)
        )
        self.secondConv = nn. Sequential(
            nn.Conv2d(25,50, kernel_size=(1,8),dilation=(1,2),padding='same'),
            nn.BatchNorm2d(50, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            get_activation(hp['act']),
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(p=0.3)
        )
        self.thirdConv = nn.Sequential(
            nn.Conv2d(50,100, kernel_size=(1,8),dilation=(1,2),padding='same'),
            nn.BatchNorm2d(100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            get_activation(hp['act']),
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(p=0.3)
        )
        self.fourthConv = nn.Sequential(
            nn.Conv2d(100,200, kernel_size=(1,8),dilation=(1,2),padding='same'),
            nn.BatchNorm2d(200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            get_activation(hp['act']),
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(p=0.3)
        )
        self.classify = nn.Linear(9200,2,bias=True)
    def forward(self,x):
        x = self.firstConv(x)
        x = self.secondConv(x)
        x = self.thirdConv(x)
        x = self.fourthConv(x)
        x = torch.flatten(x,start_dim=1,end_dim=-1)
        x = self.classify(x)
        return x
        
        


# %%
def get_loss(hp):
    return nn.CrossEntropyLoss()

def get_optimizer(hp,model):
    if hp['optimizer'] == 'sgd':
        return torch.optim.SGD(model.parameters(),lr=hp['lr'])
    elif hp['optimizer'] == 'adam':
        return torch.optim.Adam(model.parameters(),lr=hp['lr'])
def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    # turn model into training mode
    model.train()
    epoch_loss = 0.0
    correct = 0.0
    for batch, (X,y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred,y)
        epoch_loss += loss
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
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
    # turn model into eval mode
    model.eval()
    test_loss, correct = 0,0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            #print(y.dtype)
            pred = model(X)
            #print(pred.dtype)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    #print(f'Test Accuracy: {(100 * correct):>0.1f}%, test loss:{test_loss:>8f} \n')
    return correct, test_loss


# %%
def train_with_hp(hp,save_result=False):
    
   
    # get dataloader
    train_dl, test_dl = gen_loader(hp)
    # get device
    device = get_device()
    # generate model
    if hp['NN'] == "EEGNet":
        model = EEGNet(hp).to(device)
    else:
        model = DeepConvNet(hp).to(device)
    print(model)
    
    loss_fn = get_loss(hp)
    optimizer = get_optimizer(hp,model)

    test_acc_history = []
    test_loss_history = []
    train_acc_history = []
    train_loss_history = []
    
    epochs = hp['epochs']
    for t in range(epochs):
        print(f"Epoch {t+1}  ",end='')
        acc, loss = train(train_dl, model, loss_fn, optimizer,device)
        train_acc_history.append(acc)
        train_loss_history.append(loss)
        print(f"train_acc: {acc:<5.4f}  ",end='')

        acc, loss = test(test_dl, model, loss_fn, device)
        test_acc_history.append(acc)
        test_loss_history.append(loss)
        print(f"test_acc: {acc:<5.4f}  ",end='')

        print('\n',end='')

    # save model parameters
    if save_result:
        pth_name = 'checkpoints/'+hp_2_str(hp)+'model.pth'
        torch.save(model.state_dict(), pth_name)
        print(f"Save PyTorch Model State to {pth_name}")

        npz_name = 'learning_curves/'+hp_2_str(hp)+'curves.npz'
        np.savez_compressed(
            npz_name,
            train_acc=np.array(train_acc_history),
            train_loss=np.array(train_loss_history),
            test_acc=np.array(test_acc_history),
            test_loss=np.array(test_loss_history)
        )
        print(f"Save learning curves to {npz_name}")
    print("Done!")
    history = {
        'train_acc' : np.array(train_acc_history),
        'train_loss' : np.array(train_loss_history),
        'test_acc' : np.array(test_acc_history),
        'test_loss' : np.array(test_loss_history)

    }
    return model, history


def load_and_test_model(hp):
    device = get_device()

    if hp['NN'] == "EEGNet":
        model = EEGNet(hp).to(device)
    else :
        model = DeepConvNet(hp).to(device)

    hp_str = hp_2_str(hp)
    model_param_name = './checkpoints/'+hp_str+'model.pth'
    model.load_state_dict(torch.load(model_param_name))

    _, test_dl = gen_loader(hp)

    loss_fn = get_loss(hp)

    acc, loss = test(test_dl, model, loss_fn, device)
    print(f"Model:{hp['NN']} with {hp['act']}\n  {model}")
    print(f'Test ACC: {acc}')

    lh = np.load('./learning_curves/'+hp_str+'curves.npz')
    print(f"Best ACC during training: {np.max(lh['test_acc'])}")
    

# %%



if __name__ == "__main__":
    model = DeepConvNet(hp).to('cpu')
    #model = EEGNet(hp).to('cpu')
    print(model)
    train_dl , test_dl = gen_loader(hp)
    for x,y in train_dl:
        #model(x)
        print(x.shape,y.shape)
        break
    for x,y in test_dl:
        #model(x)
        print(x.shape,y.shape)
        break
    
