import torch
import torch.nn as nn
from torchvision import transforms

from unet_model import UNet


class UNet_prompt(nn.Module):
    def __init__(self, input_channel = 6, output_channel = 3):
        super(UNet_prompt, self).__init__()
        self.backbone = UNet(3,3)
        self.width = 256
        self.height = 256
        self.prompt = nn.Parameter(torch.zeros(1,3,self.width,self.height))
        self.conv = nn.Conv2d(input_channel,output_channel,1)

    def set_imageSize(self, image):
        height = image.shape[2]
        width = image.shape[3]
        setattr(self, 'width', width)
        setattr(self,'height',height)
        setattr(self, 'prompt', nn.Parameter(torch.zeros(1,3,width,height)))

    def forward(self, x):

        batch_size = x.shape[0]
        height,width = x.shape[2],x.shape[3]
        Resize = transforms.Resize((height,width))
        prompt = Resize(self.prompt)                 # 把prompt插值到和图像一样大的大小
        prompt = prompt.repeat(batch_size,1,1,1)     # 把prompt复制batch_size份
        combination = torch.cat([x,prompt],1)
        res = self.conv(combination)
        res = self.backbone(res)

        return res

if __name__ == "__main__":
    x = torch.Tensor(1,3,1024,1024)
    model = UNet_prompt()
    output = model(x)
    print(output.shape)



