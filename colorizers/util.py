
from PIL import Image
import numpy as np
from skimage import color
import torch
import torch.nn.functional as F
from IPython import embed
import matplotlib.pyplot as plt
import os
from torchvision import transforms


def load_img(img_path):
	out_np = np.asarray(Image.open(img_path))
	if(out_np.ndim==2):
		out_np = np.tile(out_np[:,:,None],3)
	return out_np

def resize_img(img, HW=(256,256), resample=3):
	return np.asarray(Image.fromarray(img).resize((HW[1],HW[0]), resample=resample))

def preprocess_img(img_rgb_orig, HW=(256,256), resample=3):
	# return original size L and resized L as torch Tensors
	img_rgb_rs = resize_img(img_rgb_orig, HW=HW, resample=resample)
	
	img_lab_orig = color.rgb2lab(img_rgb_orig)
	img_lab_rs = color.rgb2lab(img_rgb_rs)

	img_l_orig = img_lab_orig[:,:,0]
	img_l_rs = img_lab_rs[:,:,0]

	tens_orig_l = torch.Tensor(img_l_orig)[None,None,:,:]
	tens_rs_l = torch.Tensor(img_l_rs)[None,None,:,:]

	return (tens_orig_l, tens_rs_l)

def postprocess_tens(tens_orig_l, out_ab, mode='bilinear'):
	# tens_orig_l 	1 x 1 x H_orig x W_orig
	# out_ab 		1 x 2 x H x W

	HW_orig = tens_orig_l.shape[2:]
	H, W = HW_orig
	scale = float(256)/(min(H, W))
	H_tar, W_tar = int(H * scale / 8) * 8, int(W * scale / 8) * 8
	trans = transforms.Compose([transforms.Resize((H_tar, W_tar)),])
	tens_orig_l = trans(tens_orig_l)
	HW_orig = (H_tar, W_tar)
	HW = out_ab.shape[2:]

	# call resize function if needed
	if(HW_orig[0]!=HW[0] or HW_orig[1]!=HW[1]):
		out_ab_orig = F.interpolate(out_ab, size=HW_orig, mode='bilinear')
	else:
		out_ab_orig = out_ab

	out_lab_orig = torch.cat((tens_orig_l, out_ab_orig), dim=1)
	return color.lab2rgb(out_lab_orig.data.cpu().numpy()[0,...].transpose((1,2,0)))


def save_res(l, ab, ab_pred, output_dir):
	N = l.shape[0]
	for i in range(N):
		pred_img = postprocess_tens(l[i][None,...].cpu(), ab_pred[i][None,...].cpu())
		orig_img = postprocess_tens(l[i][None,...].cpu(), ab[i][None,...].cpu())
		
		plt.figure(figsize=(12,4))

		plt.subplot(1,3,1)
		plt.imshow(l[i][0], cmap='gray')
		plt.title('gray')
		plt.axis('off')

		plt.subplot(1,3,2)
		plt.imshow(orig_img)
		plt.title('original')
		plt.axis('off')

		plt.subplot(1,3,3)
		plt.imshow(pred_img)
		plt.title('predicted')
		plt.axis('off')

		plt.savefig(os.path.join(output_dir, f'res_{i}.png'))
		plt.close()
		# plt.show()

def get_metrics(y_pred, y_true):
    '''
    Calculate the metrics of gt and prediction in ab space.
    Both of shape NxHxWx3 in RGB
    '''
    N, H, W, _ = y_pred.shape
    mse = torch.tensor([F.mse_loss(torch.tensor(y_pred[i]), torch.tensor(y_true[0])) for i in range(N)])
    psnr = 10 * torch.log10(1. / mse)
    return psnr.mean()