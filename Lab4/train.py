import torch
from dataloader import get_training_pairs, get_testing_pairs, nums2word
from torch.utils import data

dl_train = data.DataLoader(get_training_pairs(),batch_size=1, shuffle=True)
dl_test = data.DataLoader(get_testing_pairs(),batch_size=1, shuffle = False)



if __name__ == '__main__':
    for w,t in dl_train:
        print(w.transpose(0,1).shape)
        print(t.shape)
        break
