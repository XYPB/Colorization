
import argparse
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms

from colorizers import *
from data import tinycoco_dataset
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('-p','--data_path', type=str, default='./dataset/COCO/')
parser.add_argument('--param_path', type=str, default='model/colorization_release_v2-9b330a0b.pth')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--output_dir', type=str, default='imgs_out/')
parser.add_argument('--num_epoch', type=int, default=10)
parser.add_argument('--img_size', type=int, default=64)

if __name__ == '__main__':
	opt = parser.parse_args()
	transformer = transforms.Compose([
        transforms.Resize(opt.img_size),
        transforms.RandomHorizontalFlip(),
    ])

	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	tr_loader = tinycoco_dataset.get_TinyCOCO_loader(root=opt.data_path, batch_size=opt.batch_size, task='train', transfomer=transformer)
	te_loader = tinycoco_dataset.get_TinyCOCO_loader(root=opt.data_path, batch_size=opt.batch_size, task='test', transfomer=transformer)
	va_loader = tinycoco_dataset.get_TinyCOCO_loader(root=opt.data_path, batch_size=opt.batch_size, task='val', transfomer=transformer)

	model = eccv16(model_path=opt.param_path).to(device)
	criteria = nn.MSELoss()
	optimizer = torch.optim.Adam([{'params':model.model1.parameters()},
									{'params':model.model2.parameters()},
									{'params':model.model3.parameters()},
									{'params':model.model4.parameters()},
									{'params':model.model5.parameters()},
									{'params':model.model6.parameters()},
									{'params':model.model7.parameters()},
									{'params':model.model8.parameters()},
									], lr=1e-4, weight_decay=1e-4)

	for epoch in range(opt.num_epoch):
		model.train()
		pbar = tqdm(tr_loader)
		total_loss = 0
		for i, (l, ab) in enumerate(pbar):
			ab.to(device)
			ab_pred = model(l.to(device))
			optimizer.zero_grad()
			loss = criteria(ab_pred.cuda(), ab.cuda())
			total_loss += loss.item()
			loss.backward()
			optimizer.step()
			pbar.set_description_str(f'Epoch {epoch}, loss: {total_loss / (i+1):.4f}')
		print(f'Finish training of {epoch} epoch...')

		model.eval()
		pbar_test = tqdm(te_loader)
		total_loss = 0
		total_cnt = len(te_loader)
		for i, (l, ab) in enumerate(pbar_test):
			ab_pred = model(l.to(device))
			total_loss += criteria(ab_pred, ab.to(device)).item
			if i == 0:
				save_res(l, ab, ab_pred, opt.output_dir)
		print(f'Test loss: {total_loss / total_cnt}')





