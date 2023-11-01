import torch
import torch.nn.functional as F
from torchvision import transforms

import PIL
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import os.path
import glob

from .modules.midas.dpt_depth import DPTDepthModel


root_dir = './third_party/omnidata/omnidata_tools/torch/pretrained_models/'

map_location = (lambda storage, loc: storage.cuda()) if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

image_size = 384
pretrained_weights_path = root_dir + 'omnidata_dpt_depth_v2.ckpt'  # 'omnidata_dpt_depth_v1.ckpt'
model = DPTDepthModel(backbone='vitb_rn50_384') # DPT Hybrid
checkpoint = torch.load(pretrained_weights_path, map_location=map_location)
if 'state_dict' in checkpoint:
    state_dict = {}
    for k, v in checkpoint['state_dict'].items():
        state_dict[k[6:]] = v
else:
    state_dict = checkpoint
model.load_state_dict(state_dict)
model.to(device)
trans_totensor = transforms.Compose([transforms.Resize(image_size, interpolation=PIL.Image.BILINEAR),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=0.5, std=0.5)])


def demo_depth_custom_func(image_dir, output_path, vis_path):
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(vis_path, exist_ok=True)

    for f in glob.glob(image_dir+'/*'):
        img_path, output_file_name = f, os.path.splitext(os.path.basename(f))[0]
        with torch.no_grad():
            save_path = os.path.join(output_path, f'{output_file_name}.npy')
            vis_path_ = os.path.join(vis_path, f'{output_file_name}.png')

            # print(f'Reading input {img_path} ...')
            
            img = Image.open(img_path)
            w, h = img.size
            resized_w = (w * 384 / h) // 64 * 64
            img = img.resize((int(resized_w), 384), PIL.Image.BILINEAR)

            img_tensor = trans_totensor(img)[:3].unsqueeze(0).to(device)

            if img_tensor.shape[1] == 1:
                img_tensor = img_tensor.repeat_interleave(3,1)

            output = model(img_tensor).clamp(min=0, max=1)

            output = F.interpolate(output.unsqueeze(0), (h, w), mode='bicubic').squeeze(0)
            output = output.clamp(0,1)
            plt.imsave(vis_path_, 1 - output.detach().cpu().squeeze(), cmap='viridis')
            np.save(save_path, output.detach().cpu().squeeze().numpy())
