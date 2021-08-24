from get_device import device
from evaluator import evaluation_model
from dataset import get_dataloader, unnormalize
from model import Generator
from train import hp_2_str, load_checkpoint
import torch
from torchvision.utils import save_image, make_grid

hp = {
    'lr': 0.001,
    'beta1': 0.0,
    'name': 'projection_scheduler_smaller_z',
    'nz': 25,  # random vector size
    'ngf': 64,  # generator features
    'ndf': 64,   # discriminator features
    'sch': True
}


netG = Generator(hp['nz'], hp['ngf']).to(device)
ckpt = load_checkpoint(hp, 'best', netG)
test_dl = get_dataloader('test', batch_size=8)
ev = evaluation_model()


imgs= []
with torch.no_grad():
    total_acc = 0
    cnt = 0
    for l in test_dl:
        l = l.to(device)
        noise = torch.randn(8, netG.nz, 1, 1, device=device)
        img = netG(noise, l.unsqueeze(-1).unsqueeze(-1))
        img_ = unnormalize(img)
        for i, (g, gl) in enumerate(zip(img_, l)):
            imgs.append(g)
            print(f'{len(imgs)}"th imges: ')
            acc = ev.eval(img[[i],:,:,:] , gl.unsqueeze(0), log=True)
            cnt += 1
            total_acc += acc
            save_image(g, f'test_{len(imgs)}.png')
    grid_img = make_grid(imgs)
    save_image(grid_img, f'test_grid_{hp_2_str(hp)}.png')
    print(f'F1-score={total_acc/cnt}')
