from sklearn.metrics import confusion_matrix, accuracy_score

import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
import numpy as np


from tqdm import tqdm

from pytorch_lightning import Trainer
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint




import argparse




class TwinResNet(LightningModule):
	def __init__(self, structure = 'resnet18'):
		super(TwinResNet, self).__init__()
		new_in_channels = 6
	#         model = models.resnet18(pretrained=True)
		model = torch.hub.load('pytorch/vision:v0.10.0', structure, pretrained=True)

		layer = model.conv1

		# Creating new Conv2d layer
		new_layer = nn.Conv2d(in_channels=new_in_channels, 
						out_channels=layer.out_channels, 
						kernel_size=layer.kernel_size, 
						stride=layer.stride, 
						padding=layer.padding,
						bias=layer.bias)

		copy_weights = 0 # Here will initialize the weights from new channel with the red channel weights

		# Copying the weights from the old to the new layer
		new_layer.weight.data[:, :layer.in_channels, :, :] = layer.weight.clone()

		#Copying the weights of the `copy_weights` channel of the old layer to the extra channels of the new layer
		for i in range(new_in_channels - layer.in_channels):
			channel = layer.in_channels + i
			new_layer.weight.data[:, channel:channel+1, :, :] = layer.weight[:, copy_weights:copy_weights+1, : :].clone()
		new_layer.weight = nn.Parameter(new_layer.weight)

		model.conv1 = new_layer
		model.fc = nn.Linear(in_features=model.fc.in_features, out_features=2, bias=True)
		self.resnet_model = model

	def forward(self, images):
		return self.resnet_model(images)

	def training_step(self, batch, batch_idx):
		logits = self(batch['images'])
		loss = F.cross_entropy(logits, batch['same'].long())
		self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
		return loss

	def validation_step(self, batch, batch_idx):
		self._shared_eval(batch, batch_idx, "val")

	def test_step(self, batch, batch_idx):
		self._shared_eval(batch, batch_idx, "test")

	def _shared_eval(self, batch, batch_idx, prefix):
		with torch.no_grad():
			logits = self(torch.cat((batch['A'], batch['B']), dim = 1))
			loss = F.cross_entropy(logits, batch['same'].long())
			acc = -accuracy_score(logits.argmax(1).cpu().numpy(), batch['same'].cpu().numpy())

		self.log(f"{prefix}_acc", acc, on_epoch=True, prog_bar=True)
		self.log(f"{prefix}_loss", loss, on_epoch=True, prog_bar=True)

	def configure_optimizers(self):
		return torch.optim.Adam(self.parameters(), lr=1e-4)


def main():
	from facedataset import MXFaceDatasetTwin, MXFaceDatasetFromBin


	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	train_set_ = MXFaceDatasetTwin(source)
	train_set = torch.utils.data.DataLoader(train_set_, batch_size = 256, shuffle = True, num_workers = 2)#, collate_fn=collate_paired_data)

	valid_set_ = MXFaceDatasetFromBin(source, 'lfw')
	valid_set = torch.utils.data.DataLoader(valid_set_, batch_size = 256, shuffle = False)

	model = TwinResNet(opt.structure)
	# model.load_state_dict(torch.load(os.path.join(source, 'classifier.pt')))

	trainer = Trainer(accelerator = 'gpu' if torch.cuda.is_available() else 'cpu',
					  gpus = [opt.device_id],
					  max_epochs = opt.max_epochs,
					  gradient_clip_val = 5,
					  callbacks = [ModelCheckpoint(monitor='val_loss')],
					  resume_from_checkpoint = opt.ckpt if len(opt.ckpt) > 5 else None
					)
	trainer.fit(model, train_set, valid_set)
	trainer.test(model, valid_set, ckpt_path = 'best')

	torch.save(model.state_dict(), os.path.join(source, 'classifier.pt'))



if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="face models")
	parser.add_argument("--source", type=str, default="faces_webface_112x112", help='path of files to process')
	parser.add_argument("--device_id", type=int, default=1, help="GPU id")
	parser.add_argument("--max_epochs", type=int, default=100, help="max epochs in training")
	parser.add_argument("--structure", type=str, default="resnet18", help='resnet models')
	parser.add_argument("--ckpt", type=str, default="", help='pytorch lightning checkp')

	opt = parser.parse_args()

	source = opt.source

	main()
