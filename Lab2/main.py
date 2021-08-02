
from nn_util import train_with_hp
# experiment 1
"""
hp = {
    'bs': 128,
    'lr': 5e-3,
    'epochs':300,
    'optimizer': 'adam', #  'sgd'  
    'NN':'EEGNet', # 'EEGNet'
    'act':'ELU' # 'ELU', 'LeakyReLU'
}

# experiment 2

hp = {
    'bs': 128,
    'lr': 1e-2,
    'epochs':300,
    'optimizer': 'adam', #  'sgd'  
    'NN':'EEGNet', # 'EEGNet'
    'act':'ELU' # 'ELU', 'LeakyReLU'
}
"""

# experiment 3

hp = {
    'bs': 128,
    'lr': 1e-3,
    'epochs':350,
    'optimizer': 'adam', #  'sgd'  
    'NN':'EEGNet', # 'EEGNet'
    'act':'ELU' # 'ELU', 'LeakyReLU'
}

if __name__ == '__main__':
    for m in ['DeepConvNet','EEGNet']:
        for act in ['ELU','ReLU','LeakyReLU']:
            hp['NN'] = m
            hp['act'] = act
            model , hist = train_with_hp(hp,save_result=True)
