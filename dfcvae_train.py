import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
import numpy as np

from facedataset import MXFaceDatasetConventional, MXFaceDatasetBalancedIntraInterClusters, collate_paired_data, MXFaceDatasetFromBin
from tqdm import tqdm

from pytorch_lightning import Trainer
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint


import argparse


import sys
sys.path.append('PyTorch-VAE')

from models import DFCVAE
from torchviz import make_dot


def main():
	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	# train_set_ = MXFaceDatasetConventional(source)
	# train_set = torch.utils.data.DataLoader(train_set_, batch_size = 512, shuffle = True, num_workers = 2, collate_fn=None)

	train_set_ = MXFaceDatasetBalancedIntraInterClusters(source, resize = 64)
	train_set = torch.utils.data.DataLoader(train_set_, batch_size = 64, shuffle = True, num_workers = 2, collate_fn=collate_paired_data)

	valid_set_ = MXFaceDatasetFromBin(source, 'lfw', resize = 64)
	valid_set = torch.utils.data.DataLoader(valid_set_, batch_size = 64, shuffle = False)

	model = DFCVAE(3, 512)    
	# model.load_state_dict(torch.load('drive/MyDrive/Colab Notebooks/Face-Data/vae.pt', map_location=device))

	# sample_output = model(next(iter(train_set))['images'])[0]
	# make_dot(sample_output, params=dict(list(model.named_parameters())), show_attrs=True, show_saved=True).render("dfcvae")

	trainer = Trainer(accelerator = 'gpu' if torch.cuda.is_available() else 'cpu',
					  gpus = [opt.device_id],
					  max_epochs = 100,
					  gradient_clip_val = 1.5,
					  callbacks = [ModelCheckpoint(monitor='val_loss')],
# 					  resume_from_checkpoint = "path/to/ckpt/file/checkopoint.ckpt"
	)
	trainer.fit(model, train_set, valid_set)

	trainer.test(model, valid_set, ckpt_path='best')

	torch.save(model.state_dict(), os.path.join(source, 'dfcvae.pt'))


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="face models")
	parser.add_argument("--source", type=str, default="faces_webface_112x112", help='path of files to process')
	parser.add_argument("--device_id", type=int, default=1, help="GPU id")

	opt = parser.parse_args()

	source = opt.source

	main()
