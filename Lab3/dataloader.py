import pandas as pd
from torch.utils import data
import numpy as np
from os.path import join as join
import torch
import matplotlib.image as mpimg
from torchvision import transforms

def getData(mode):
    if mode == 'train':
        img = pd.read_csv('train_img.csv')
        label = pd.read_csv('train_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)
    else:
        img = pd.read_csv('test_img.csv')
        label = pd.read_csv('test_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)


class RetinopathyLoader(data.Dataset):
    def __init__(self, root, mode):
        """
        Args:
            root (string): Root path of the dataset.
            mode : Indicate procedure status(training or testing)

            self.img_name (string list): String list that store all image names.
            self.label (int or float list): Numerical list that store all ground truth label values.
        """
        self.root = root
        self.img_name, self.label = getData(mode)
        self.mode = mode
        print("> Found %d images..." % (len(self.img_name)))
        self.transform = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)

    def __getitem__(self, index):
        """something you should implement here"""

        """
           step1. Get the image path from 'self.img_name' and load it.
                  hint : path = root + self.img_name[index] + '.jpeg'

           step2. Get the ground truth label from self.label

           step3. Transform the .jpeg rgb images during the training phase, such as resizing, random flipping,
                  rotation, cropping, normalization etc. But at the beginning, I suggest you follow the hints.

                  In the testing phase, if you have a normalization process during the training phase, you only need
                  to normalize the data.

                  hints : Convert the pixel value to [0, 1]
                          Transpose the image shape from [H, W, C] to [C, H, W]

            step4. Return processed image and label
        """
        img_path = self.img_name[index]
        label = self.label[index]

        img = mpimg.imread(join(self.root, img_path)+'.jpeg')
        #print(f'pixel_max = {np.max(img)}')

        img_t = torch.tensor(img)
        # convert from uint8 to float
        img_t = img_t.type(torch.FloatTensor)
        # change the range to [0,1]
        img_t /= 255.0
        # Transpose from shape [H,W,C] to [C, H, W]
        img_t = torch.transpose(img_t,0,2)
        # normalize image
        if self.transform:
            img_t = self.transform(img_t)

        return img_t , label

def get_loader(hp):
    train_dl = data.DataLoader(
        RetinopathyLoader('data','train'),
        shuffle=True,
        batch_size=hp['bs'],
        num_workers=4
    )
    test_dl = data.DataLoader(
        RetinopathyLoader('data','test'),
        shuffle=False,
        batch_size=hp['bs'],
        num_workers=4
    )
    return train_dl, test_dl
