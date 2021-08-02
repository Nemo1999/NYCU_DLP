from nn_util import hp_2_str
import numpy as np
import matplotlib.pyplot as plt 
from main import hp
models = ['EEGNet','DeepConvNet']

def plot_learning_curves(model,hp):
    plt.figure(figsize=(10,8))
    hp['NN'] = model
    activations = ['ELU','LeakyReLU', 'ReLU']
    for act in activations:
        hp['act'] = act
        lh = np.load('./learning_curves/'+hp_2_str(hp)+'curves.npz')
        plt.plot(lh['train_acc'],label=f'{act}-train',linewidth=1)
        plt.plot(lh['test_acc'],label=f'{act}-test',linewidth=1)
        print(f'Final ACC for {model} using {act:<17}: {lh["test_acc"][-1]:>7.3f}')
    plt.xlabel('epochs')
    plt.ylabel('acc')
    plt.yticks(np.arange(0.6,1.0,0.05))
    plt.title(f'Activation Function Comparason ({model})',fontsize=20)
    plt.legend()
    plt.savefig(f'Compare_{hp["NN"]}.png',backend='agg')
if __name__ == '__main__':
    for i in range(2):
        plot_learning_curves(models[i],hp)
