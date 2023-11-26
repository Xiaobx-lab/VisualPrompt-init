import numpy as np
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

from dataset import RESIDE_Dataset
from initialization import UNet_prompt
from option import batch_size, lr, steps, path, model_dir
from utils import ssim, psnr


def test(net, loader_test):
    net.eval()
    torch.cuda.empty_cache()
    ssims = []
    psnrs = []
    for i in range(100):
        input, label = next(iter(loader_test))
        input = input.to(device)
        label = label.to(device)
        pred = net(input)  # 预测值
        ssim1 = ssim(pred, label).item()  # 真实值
        psnr1 = psnr(pred, label)
        ssims.append(ssim1)
        psnrs.append(psnr1)
        # 输出图片看一下
        save_image(input,'input.jpg')
        save_image(label, 'label.jpg')
        save_image(pred, 'predict.jpg')

    return np.mean(ssims), np.mean(psnrs)


def train(loader_train, froze_flag, loader_test, net, optimizer, criterion):

    #  ------------- 冻结模型中Unet部分的参数 ---------------
    if froze_flag:
        for k,v in net.named_modules():
            if 'backbone' in k:
                v.requires_grad = False
    #  ------------------------------------------------

    max_ssim = 0
    max_psnr = 0
    ssims = []
    psnrs = []
    print('train begin!')
    for step in range(steps):
        degrade, label = next(iter(loader_train))
        degrade = degrade.to(device)
        label = degrade.to(device)
        net = net.to(device)
        pre = net(degrade)
        loss = criterion(pre,label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f'\rtrain loss : {loss.item():.5f}| step :{step}/{steps}|lr :{lr :.7f} ',end='', flush=True)
    # 测试下
        if (step+1) % 500 == 0:
            print('\n TEST !')
            with torch.no_grad():
                ssim_eval, psnr_eval = test(net, loader_test)
                print(f'\n step :{step} |ssim:{ssim_eval:.4f}| psnr:{psnr_eval:.4f}')
                ssims.append(ssim_eval)
                psnrs.append(psnr_eval)
                if ssim_eval > max_ssim and psnr_eval > max_psnr:
                    max_ssim = max(max_ssim, ssim_eval)
                    max_psnr = max(max_psnr, psnr_eval)
                torch.save(net.state_dict(), model_dir+'/train_model.pth')
                print(f'\n model saved at step :{step}|\n save path: {model_dir}/train_model.pth | max_psnr:{max_psnr:.4f}|max_ssim:{max_ssim:.4f}')




device = 'cuda' if torch.cuda.is_available() else 'cpu'
train_loader = DataLoader(dataset=RESIDE_Dataset(path , train=True), batch_size=batch_size,shuffle=True)
test_loader = DataLoader(dataset=RESIDE_Dataset(path + '/test/', train=False), batch_size=batch_size,shuffle=True)

net = UNet_prompt().to(device)
optimizer = optim.Adam(params=filter(lambda x: x.requires_grad, net.parameters()), lr=lr, betas=(0.9, 0.999),
                       eps=1e-08)

froze_flag = False
criterion = nn.L1Loss().to(device)
optimizer.zero_grad()

train(train_loader, froze_flag, test_loader, net, optimizer, criterion)
