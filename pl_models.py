import sys

from pytorch_lightning.core.lightning import LightningModule

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

import numpy as np
from sklearn.metrics import accuracy_score


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

sys.path.append('insightface/recognition/arcface_torch')
from backbones import get_model as insf_get_model
sys.path.pop()

sys.path.append('arcface-pytorch/models')
from metrics import ArcMarginProduct
sys.path.pop()

sys.path.append('multiple-attention')
from models.MAT import MAT
sys.path.pop()

from pytorchcv.model_provider import get_model as ptcv_get_model

def kimera_get_model(structure, num_classes):
	if structure in ('resnet34', 'resnet50', 'resnet101', 'resnet152'):

		m = torch.hub.load('pytorch/vision:v0.10.0', structure, pretrained=True)

		if num_classes == m.fc.in_features:
			return FeatureExtractor(m)
		else:
			m.fc = nn.Linear(in_features=m.fc.in_features, out_features=num_classes)
			return m

	elif structure.startswith('multiatt'):
		m = MAT(structure.replace('multiatt', ''))
		m.ensemble_classifier_fc[2] = nn.Linear(in_features=m.ensemble_classifier_fc[2].in_features, out_features=num_classes, bias=True)
		return m

	elif structure in ('r18', 'r34', 'r50', 'r100'):
		return insf_get_model(structure, num_features = num_classes)

	else:
		m = ptcv_get_model(structure, pretrained=True)
		m.features.final_pool = nn.AdaptiveAvgPool2d((1, 1))
		m.output = nn.Linear(in_features=m.output.in_features, out_features=num_classes, bias=True)
		return m

class ArcFace(LightningModule):
	def __init__(self, out_features = 13938, embeddings = 512, structure = 'resnet18'):
		super(ArcFace, self).__init__()
		self.backbone = kimera_get_model(structure = structure, num_classes = embeddings)
		self.header = ArcMarginProduct(in_features=embeddings, out_features=out_features, s=30, m=0.5)

	def extract_feature(self, images):
		return self.backbone(images)

	def forward(self, images, labels):
		features = self.extract_feature(images)
		return self.header(features, labels)

	def training_step(self, batch, batch_idx):
		logits = self(batch['images'], batch['person_ids'])
		loss = F.cross_entropy(logits, batch['person_ids'])
		self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
		return loss

	def validation_step(self, batch, batch_idx):
		self._shared_eval(batch, batch_idx, "val")

	def test_step(self, batch, batch_idx):
		self._shared_eval(batch, batch_idx, "test")

	def _shared_eval(self, batch, batch_idx, prefix):
		with torch.no_grad():
			feat = self.extract_feature(torch.cat([batch['A'], batch['B']], dim = 0))
			cos = F.cosine_similarity(feat[:len(feat) //2], feat[len(feat) //2:])
			angles = torch.acos(cos).cpu().numpy()

		y = batch['same'].cpu().numpy()
		y_pred = (angles - np.radians(75)) > 0
		acc = -accuracy_score(y, y_pred)

		self.log(f"{prefix}_acc", acc, on_epoch=True, prog_bar=True)  

	def configure_optimizers(self):
		return torch.optim.SGD(self.parameters(), lr=1e-1, momentum=0.9, weight_decay=5e-4)


sys.path.append('face-parsing.PyTorch')
from model import BiSeNet
face_parser = BiSeNet(n_classes=19)
sys.path.pop()

class ArcOrgans(ArcFace):
	def __init__(self, out_features = 13938, embeddings = 512, structure = 'r18'):
		super(ArcOrgans, self).__init__()

		self.face_parser = face_parser.eval()
		self.face_parser.load_state_dict(torch.load('79999_iter.pth', map_location = 'cpu'))
		for param in self.face_parser.parameters():
			param.requires_grad = False

	def extract_feature(self, images):
		with torch.no_grad():
			masks = self.face_parse(images)
		images = torch.matmul(images.permute(0, 2, 3, 1).unsqueeze(-1), masks.permute(0, 2, 3, 1).unsqueeze(-2).float()).permute(4, 0, 3, 1, 2)[-1]
		return self.backbone(images)

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

	def configure_optimizers(self):
		return torch.optim.Adam(self.parameters(), lr=1e-4)

class ArcSegFace(ArcOrgans):
	def __init__(self, out_features = 13938, embeddings = [102, 102, 102, 102, 104], structures = ['resnet10', 'resnet10', 'resnet10', 'resnet10', 'resnet10']):
		super(ArcSegFace, self).__init__()

		self.backbone = None
		self.backbones = nn.ModuleList([kimera_get_model(structure = struct, num_classes = emb) for emb, struct in zip(embeddings, structures)])

	def extract_feature(self, images):
		with torch.no_grad():
			masks = self.face_parse(images)
		images = torch.matmul(images.permute(0, 2, 3, 1).unsqueeze(-1), masks.permute(0, 2, 3, 1).unsqueeze(-2).float()).permute(4, 0, 3, 1, 2)
		images = images[:len(self.backbones)]
		return torch.cat([b(img) for b, img in zip(self.backbones, images)], dim = 1)

