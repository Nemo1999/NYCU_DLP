import json
import torch
from torch.utils import data
from torchvision import transforms
from matplotlib.pyplot import imread as pimread
import os
import numpy as np
from get_device import device
from torchvision.utils import save_image

def get_iCLEVR_data(root_folder,mode):
    if mode == 'train':
        data = json.load(open(os.path.join(root_folder,'train.json')))
        obj = json.load(open(os.path.join(root_folder,'objects.json')))
        img = list(data.keys())
        label = list(data.values())
        for i in range(len(label)):
            for j in range(len(label[i])):
                label[i][j] = obj[label[i][j]]
            tmp = np.zeros(len(obj), dtype=np.float)
            tmp[label[i]] = 1
            label[i] = tmp
        return np.squeeze(img), np.squeeze(label)
    else:
        data = json.load(open(os.path.join(root_folder,'test.json')))
        obj = json.load(open(os.path.join(root_folder,'objects.json')))
        label = data
        for i in range(len(label)):
            for j in range(len(label[i])):
                label[i][j] = obj[label[i][j]]
            tmp = np.zeros(len(obj), dtype=np.float)
            tmp[label[i]] = 1
            label[i] = tmp
        return None, label


class ICLEVRLoader(data.Dataset):
    def __init__(self, root_folder, mode='train'):
        self.root_folder = root_folder
        self.mode = mode
        self.trans = transforms.Compose([
            transforms.Resize(size=(64, 64), antialias=True),
            transforms.RandomCrop(size=64, padding=None),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.img_list, self.label_list = get_iCLEVR_data(root_folder,mode)

        self.num_classes = 24
    def __len__(self):
        """'return the size of dataset"""
        return len(self.label_list)

    def __getitem__(self, index):
        if self.mode == 'train':
            img_file = self.img_list[index]
            img_path = os.path.join(self.root_folder, 'images', img_file )
            img = pimread(img_path)
            # get rid off alpha channel and permute to (ch, H , W)
            # change color space from RGB to BGR
            img = torch.tensor(img[:, :, [0, 1, 2]]).permute(2, 0, 1)

            if self.trans:
                img = self.trans(img)
            #print(img.shape)
            #print(torch.max(img), torch.min(img), img.dtype)
            img = img.float()
            lab = self.label_list[index]
            lab = torch.tensor(lab, dtype=torch.float)
            return img, lab
        else:
            return torch.tensor(self.label_list[index],dtype=torch.float)

def get_dataloader(mode, root_folder='/home/nemo/Labs/Lab5', batch_size=4):
    if mode == 'train':
        dataset = ICLEVRLoader(root_folder, mode)
        return data.DataLoader(dataset, batch_size, shuffle=True, num_workers=4)
    else:
        dataset = ICLEVRLoader(root_folder, mode)
        return data.DataLoader(dataset, batch_size, shuffle=False)

unnorm = transforms.Normalize((-1, -1, -1), (2, 2, 2))
def unnormalize(img):
    return unnorm(img)


if __name__ == '__main__':

    from evaluator import evaluation_model
    train_dl = get_dataloader('train', batch_size=16)
    acc_total = 0
    total = 0
    ev = evaluation_model()
    for cnt, (img, lab) in enumerate(train_dl):
        # img = transforms.Resize(size=(64, 64), antialias=True)(img)
        img = img.to(device)
        lab = lab.to(device)
        acc = ev.eval(img, lab)
        """
        if acc < 1.0:
            img_unnormalized = transforms.Normalize((-1, -1, -1), (2, 2, 2,))(img)
            print(torch.max(img_unnormalized), torch.min(img_unnormalized))
            save_image(img_unnormalized, 'load_image.png')
        """
        print(acc)
        acc_total += acc
        total += 1
        if total == 100:
            ev.eval(img, lab, log=True)
            print(f'label: {lab}')
            break
    print('total',  acc_total /total)
