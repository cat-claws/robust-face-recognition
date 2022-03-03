from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC
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

# Throw away the layer that does classification on this model (the last layer).
# Then discard the height and width dimensions using tensor.view.
class FeatureExtractor(nn.Module):
	def __init__(self, resnet_model):
		super(FeatureExtractor, self).__init__()
		# remove the last layer
		self.truncated_resnet = nn.Sequential(*list(resnet_model.children())[:-1])
	def forward(self, x):
		feats = self.truncated_resnet(x)
		return feats.view(feats.size(0), -1)

class ArcMarginProduct(nn.Module):
	r"""Implement of large margin arc distance: :
		Args:
			in_features: size of each input sample
			out_features: size of each output sample
			s: norm of input feature
			m: margin
			cos(theta + m)
		"""
	def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
		super(ArcMarginProduct, self).__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.s = s
		self.m = m
		self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
		nn.init.xavier_uniform_(self.weight)

		self.easy_margin = easy_margin
		self.cos_m = math.cos(m)
		self.sin_m = math.sin(m)
		self.th = math.cos(math.pi - m)
		self.mm = math.sin(math.pi - m) * m

	def forward(self, input, label):
		# --------------------------- cos(theta) & phi(theta) ---------------------------
		cosine = F.linear(F.normalize(input), F.normalize(self.weight))
		sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
		phi = cosine * self.cos_m - sine * self.sin_m
		if self.easy_margin:
			phi = torch.where(cosine > 0, phi, cosine)
		else:
			phi = torch.where(cosine > self.th, phi, cosine - self.mm)
		# --------------------------- convert label to one-hot ---------------------------
		one_hot = torch.zeros(cosine.size()).type_as(label)
		one_hot.scatter_(1, label.view(-1, 1).long(), 1)
		# -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
		output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
		output *= self.s

		return output


class ArcFace(LightningModule):
	def __init__(self, out_features = 13938, embeddings = 512, structure = 'resnet18'):
		super(ArcFace, self).__init__()
		resnet_model = torch.hub.load('pytorch/vision:v0.10.0', structure, pretrained=True)
		
		if embeddings == resnet_model.fc.in_features:
			self.backbone = FeatureExtractor(resnet_model)
			self.header = ArcMarginProduct(in_features=embeddings, out_features=out_features, s=30, m=0.5)
		else:
			self.backbone = resnet_model
			self.backbone.fc = nn.Linear(in_features=resnet_model.fc.in_features, out_features=embeddings)
			self.header = ArcMarginProduct(in_features=embeddings, out_features=out_features, s=30, m=0.5)

	def forward(self, images, labels):
		features = self.backbone(images)
		return self.header(features, labels)

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
			feat = self.backbone(torch.cat([batch['A'], batch['B']], dim = 0))
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

	resnet_model = torch.hub.load('pytorch/vision:v0.10.0', opt.structure, pretrained=True)

	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	# train_set_ = MXFaceDatasetConventional(source)
	# train_set = torch.utils.data.DataLoader(train_set_, batch_size = 512, shuffle = True, num_workers = 2, collate_fn=None)

	train_set_ = MXFaceDatasetConventional(source)
	train_set = torch.utils.data.DataLoader(train_set_, batch_size = 512, shuffle = True, num_workers = 2)#, collate_fn=collate_paired_data)

	valid_set_ = MXFaceDatasetFromBin(source, 'lfw')
	valid_set = torch.utils.data.DataLoader(valid_set_, batch_size = 1200, shuffle = False)

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

	torch.save(model.state_dict(), os.path.join(source, 'arcface.pt'))


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
	parser.add_argument("--ckpt", type=str, default="resnet18", help='pytorch lightning checkp')
	parser.add_argument("--source", type=str, default="faces_webface_112x112", help='path of files to process')
	parser.add_argument("--num_persons", type=int, default=13938, help="number of persons in training set")
	parser.add_argument("--embedding_dim", type=int, default=512, help="embedding dimension")
	parser.add_argument("--device_id", type=int, default=1, help="GPU id")

	opt = parser.parse_args()

	num_persons = opt.num_persons
	source = opt.source
	
	main()
