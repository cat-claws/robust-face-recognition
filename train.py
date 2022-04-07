from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, accuracy_score

import numpy as np
from tqdm import tqdm

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

import sys
import argparse

import pl_models

def main():	
		
	from facedataset import MXFaceDatasetConventional, MXFaceDatasetBalancedIntraInterClusters, collate_paired_data, MXFaceDatasetFromBin

	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	train_set_ = MXFaceDatasetConventional(opt.source)
	train_set = torch.utils.data.DataLoader(train_set_, batch_size = opt.batch_size, shuffle = True, num_workers = 2)#, collate_fn=collate_paired_data)

	valid_set_ = MXFaceDatasetFromBin(opt.source, opt.valid_source)
	valid_set = torch.utils.data.DataLoader(valid_set_, batch_size = opt.batch_size, shuffle = True, num_workers = 2)


	if opt.select == 'arcface':
		model = pl_models.ArcFace(out_features = opt.num_persons, embeddings = opt.embedding_dim, structure = opt.structure)
	elif opt.select == 'arcorgans':
		model = pl_models.ArcOrgans(out_features = opt.num_persons, embeddings = opt.embedding_dim, structure = opt.structure)
	elif opt.select == 'arcsegface':
		embedding_dims = [int(x) for x in opt.embedding_dims.split(',')]
		structures = opt.structures.split(',')
		model = pl_models.ArcSegFace(out_features = opt.num_persons, embeddings = embedding_dims, structures = structures)

	# model.load_state_dict(torch.load(os.path.join(opt.source, 'arcface.pt')))
	model.header.load_state_dict({k.replace('header.', ''):v for k, v in torch.load('faces_webface_112x112/arcface.pt', map_location = 'cpu').items() if k.replace('header.', '') in model.header.state_dict()})

	trainer = Trainer(accelerator='gpu' if torch.cuda.is_available() else 'cpu',
			  gpus=[opt.device_id],
			  max_epochs=opt.max_epochs,
			  gradient_clip_val=5,
			  callbacks=[ModelCheckpoint(monitor="val_acc")],
			 )
	trainer.fit(model, train_set, valid_set, ckpt_path=opt.ckpt if len(opt.ckpt) > 5 else None)
	trainer.test(model, valid_set)#, ckpt_path = '/home/ruihan/facereco/lightning_logs/version_13/checkpoints/epoch=3-step=40887.ckpt')

	# torch.save(model.state_dict(), os.path.join(opt.source, f'{opt.select}.pt'))


	model = model.to(device)
	model.eval()
	features_ = []
	for batch in tqdm(valid_set):
		ids = batch['id']
		same = batch['same']
		with torch.no_grad():
			feat = model.extract_feature(torch.cat([batch['A'], batch['B']], dim = 0).to(device))
			features_.append((ids, feat[:len(feat) //2], feat[len(feat) //2:], same))

	features = [torch.cat(x) for x in zip(*features_)]

	angles = torch.acos(torch.nn.functional.cosine_similarity(features[1], features[2])).cpu().numpy()
	y = features[3].cpu().numpy()

	svc = LinearSVC()
	svc.fit(np.expand_dims(angles, 1), y)
	y_pred = svc.predict(np.expand_dims(angles, 1))
	print(confusion_matrix(y, y_pred))
	print(accuracy_score(y, y_pred))


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="face models")
	parser.add_argument("--select", type=str, default="arcface", help='which model to train')
	parser.add_argument("--max_epochs", type=int, default=10, help="max epochs in training")
	parser.add_argument("--batch_size", type=int, default=48, help="batch size in training")
	parser.add_argument("--ckpt", type=str, default="", help='pytorch lightning checkpoint')
	parser.add_argument("--source", type=str, default="faces_webface_112x112", help='directory of files to process')
	parser.add_argument("--valid_source", type=str, default="lfw", help='path of validation files')
	parser.add_argument("--num_persons", type=int, default=13938, help="number of persons in training set")
	parser.add_argument("--embedding_dim", type=int, default=512, help="embedding dimension")
	parser.add_argument("--structure", type=str, default="resnet18", help='resnet models')
	parser.add_argument('--embedding_dims', help='delimited list input for embeddings', type=str)
	parser.add_argument('--structures', help='delimited list input for structures', type=str, default="resnet10,resnet10,resnet10,resnet10,resnet10")
	parser.add_argument("--device_id", type=int, default=1, help="GPU id")

	opt = parser.parse_args()
	
	main()