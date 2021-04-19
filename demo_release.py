
import argparse
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from PIL import Image

from colorizers import *


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--img_path', type=str, default='./torate')
parser.add_argument('--use_gpu', action='store_true',
                    help='whether to use GPU')
parser.add_argument('--param_path', type=str,
                    default='model/colorization_release_v2-9b330a0b.pth')
parser.add_argument('-o', '--save_prefix', type=str, default='saved',
                    help='will save into this file with {eccv16.png, siggraph17.png} suffixes')
opt = parser.parse_args()

# load colorizers
colorizer_eccv16 = eccv16(pretrained=True, model_path=opt.param_path).eval()
colorizer_siggraph17 = siggraph17(pretrained=True).eval()
if(opt.use_gpu):
    colorizer_eccv16.cuda()
    colorizer_siggraph17.cuda()

# default size to process images is 256x256
# grab L channel in both original ("orig") and resized ("rs") resolutions
for img_name in tqdm(os.listdir(opt.img_path)):
    img = load_img(os.path.join(opt.img_path, img_name))

    (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256, 256))
    if(opt.use_gpu):
        tens_l_rs = tens_l_rs.cuda()

    # colorizer outputs 256x256 ab map
    # resize and concatenate to original L channel
    img_bw = postprocess_tens(tens_l_orig, torch.cat(
        (0*tens_l_orig, 0*tens_l_orig), dim=1))
    out_img_eccv16 = postprocess_tens(
        tens_l_orig, colorizer_eccv16(tens_l_rs).cpu())
    out_img_siggraph17 = postprocess_tens(
        tens_l_orig, colorizer_siggraph17(tens_l_rs).cpu())
    plt.imsave(os.path.join('eccv16/', img_name), out_img_eccv16)
    plt.imsave(os.path.join('siggraph/', img_name), out_img_siggraph17)

    # plt.imsave('%s_eccv16.png'%opt.save_prefix, out_img_eccv16)
    # plt.imsave('%s_siggraph17.png'%opt.save_prefix, out_img_siggraph17)

    # plt.figure(figsize=(12,4))
    # plt.subplot(1,3,1)
    # plt.imshow(img)
    # plt.title('Original')
    # plt.axis('off')

    # plt.subplot(1,3,2)
    # plt.imshow(img_bw)
    # plt.title('Input')
    # plt.axis('off')

    # plt.subplot(1,3,3)
    # plt.imshow(out_img_eccv16)
    # plt.title('Output (ECCV 16)')
    # plt.axis('off')
    # img_name = opt.img_path.split('/')[-1]
    # plt.savefig(os.path.join('./imgs_out', 'eccv_' + img_name))
    # plt.show()
