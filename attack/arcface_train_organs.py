from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, accuracy_score

import os
import sys
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

sys.path.append('face-parsing.PyTorch')
from model import BiSeNet
face_parser = BiSeNet(n_classes=19)
sys.path.pop()

sys.path.append('insightface/recognition/arcface_torch')
sys.path.append('arcface-pytorch/models')

from backbones import get_model
from metrics import ArcMarginProduct

import argparse

class ArcFace(LightningModule):
	def __init__(self, out_features = 13938, embeddings = 512, structure = 'resnet18'):
		super(ArcFace, self).__init__()
		if structure.startswith('resnet'):
			resnet_model = torch.hub.load('pytorch/vision:v0.10.0', structure, pretrained=True)

			if embeddings == resnet_model.fc.in_features:
				self.backbone = FeatureExtractor(resnet_model)
			else:
				self.backbone = resnet_model
				self.backbone.fc = nn.Linear(in_features=resnet_model.fc.in_features, out_features=embeddings)
		else:
			self.backbone = get_model(structure)
		self.header = ArcMarginProduct(in_features=embeddings, out_features=out_features, s=30, m=0.5)
		self.face_parser = face_parser.eval()
		self.face_parser.load_state_dict(torch.load('79999_iter.pth', map_location = 'cpu'))
		for param in self.face_parser.parameters():
			param.requires_grad = False


	def forward(self, images, labels):
		with torch.no_grad():
			masks = self.face_parse(images)
			images = torch.matmul(images.permute(0, 2, 3, 1).unsqueeze(-1), masks.permute(0, 2, 3, 1).unsqueeze(-2).float()).permute(4, 0, 3, 1, 2)[-1]
		features = self.backbone(images)
		return self.header(features, labels)
	

	def face_parse(self, x):
		x_ = T.Resize((512, 512))(x)
		mask = F.one_hot(self.face_parser(x_)[0].argmax(1), 19).permute(0, 3, 1, 2)
		mask = T.Resize(x.shape[-2:], interpolation = T.InterpolationMode.NEAREST)(mask)

		mask[:, 0] = mask[:, 4:6].sum(1)
		mask[:, 1] = mask[:, 2:4].sum(1)
		mask[:, 2] = mask[:, 10:11].sum(1)
		mask[:, 3] = mask[:, 11:14].sum(1)

		mask[:, 4] = mask[:, :4].sum(1)
		return mask[:,:5]
		

	def training_step(self, batch, batch_idx):
		logits = self(batch['images'], batch['person_ids'])
		loss = F.cross_entropy(logits, batch['person_ids'])
		return loss

	def validation_step(self, batch, batch_idx):
		self._shared_eval(batch, batch_idx, "val")

	def test_step(self, batch, batch_idx):
		self._shared_eval(batch, batch_idx, "test")

	def _shared_eval(self, batch, batch_idx, prefix):
		with torch.no_grad():
			images = torch.cat([batch['A'], batch['B']], dim = 0)
			masks = self.face_parse(images)
			images = torch.matmul(images.permute(0, 2, 3, 1).unsqueeze(-1), masks.permute(0, 2, 3, 1).unsqueeze(-2).float()).permute(4, 0, 3, 1, 2)[-1]
			feat = self.backbone(images)
			feat = feat.cpu().numpy()

			FA = normalize(feat[:len(feat) //2])
			FB = normalize(feat[len(feat) //2:])

		distances = np.array([np.linalg.norm(a - b, axis=0) for a, b in zip(FA, FB)])
		y = batch['same'].cpu().numpy()

		svc = LinearSVC()
		svc.fit(np.expand_dims(distances, 1), y)
		y_pred = svc.predict(np.expand_dims(distances, 1))
		acc = -accuracy_score(y, y_pred)
		self.log(f"{prefix}_acc", acc, on_epoch=True, prog_bar=True)  

	def configure_optimizers(self):
		# return  torch.optim.SGD(self.parameters(), lr = 5e-2, momentum = 0.9, weight_decay = 1e-4)
		return torch.optim.SGD(self.parameters(), lr=1e-1, momentum=0.9, weight_decay=5e-4)
		# return torch.optim.Adam(self.parameters(), lr=1e-4)

def main():
	
	
		
	from facedataset import MXFaceDatasetConventional, MXFaceDatasetBalancedIntraInterClusters, collate_paired_data, MXFaceDatasetFromBin

	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	# train_set_ = MXFaceDatasetConventional(source)
	# train_set = torch.utils.data.DataLoader(train_set_, batch_size = 512, shuffle = True, num_workers = 2, collate_fn=None)

	train_set_ = MXFaceDatasetConventional(source)
	train_set = torch.utils.data.DataLoader(train_set_, batch_size = 48, shuffle = True, num_workers = 2)#, collate_fn=collate_paired_data)

	valid_set_ = MXFaceDatasetFromBin(source, 'lfw')
	valid_set = torch.utils.data.DataLoader(valid_set_, batch_size = 48, shuffle = True, num_workers = 2)

	model = ArcFace(out_features = num_persons, embeddings = opt.embedding_dim, structure = opt.structure)
	# model.load_state_dict(torch.load(os.path.join(source, 'arcface.pt')))

	trainer = Trainer(accelerator='gpu' if torch.cuda.is_available() else 'cpu',
			  gpus=[opt.device_id],
			  max_epochs=opt.max_epochs,
			  gradient_clip_val=5,
			  callbacks=[ModelCheckpoint(monitor="val_acc")],
			  resume_from_checkpoint = opt.ckpt if len(opt.ckpt) > 5 else None
			 )
	trainer.fit(model, train_set, valid_set)
	trainer.test(model, valid_set, ckpt_path = 'best')

	torch.save(model.state_dict(), os.path.join(source, 'arcface_fork.pt'))


	model = model.to(device)
	model.eval()
	features_ = []
	for batch in tqdm(valid_set):
		ids = batch['id'].cpu().numpy()
		same = batch['same'].cpu().numpy()
		with torch.no_grad():
			feat = model.backbone(torch.cat([batch['A'], batch['B']], dim = 0).to(device))
			feat = feat.cpu().numpy()
			features_.append((ids, feat[:len(feat) //2], feat[len(feat) //2:], same))

	features = [np.concatenate(x) for x in zip(*features_)]


	distances = np.array([np.linalg.norm((normalize(a.reshape(1, -1)) - normalize(b.reshape(1, -1))).squeeze(0), axis=0) for a, b in zip(*features[1:3])])
	y = features[-1]

	svc = LinearSVC()
	svc.fit(np.expand_dims(distances, 1), y)
	y_pred = svc.predict(np.expand_dims(distances, 1))
	print(confusion_matrix(y, y_pred))


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="face models")
	parser.add_argument("--max_epochs", type=int, default=100, help="max epochs in training")
	parser.add_argument("--structure", type=str, default="resnet18", help='resnet models')
	parser.add_argument("--ckpt", type=str, default="", help='pytorch lightning checkp')
	parser.add_argument("--source", type=str, default="faces_webface_112x112", help='path of files to process')
	parser.add_argument("--num_persons", type=int, default=13938, help="number of persons in training set")
	parser.add_argument("--embedding_dim", type=int, default=512, help="embedding dimension")
	parser.add_argument("--device_id", type=int, default=1, help="GPU id")

	opt = parser.parse_args()

	num_persons = opt.num_persons
	source = opt.source
	
	main()
