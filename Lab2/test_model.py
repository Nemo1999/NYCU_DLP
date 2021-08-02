from main import hp
from nn_util import load_and_test_model

if __name__ == '__main__':
    for m in ['EEGNet','DeepConvNet']:
        for act in ['ELU','ReLU','LeakyReLU']:
            hp['NN'] = m
            hp['act'] = act
            load_and_test_model(hp)

