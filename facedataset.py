import os
import torch
import numpy as np
import mxnet as mx

import torch.nn.functional as F
import torchvision.transforms as T
# torch.manual_seed(1234)


def get_person_id_category(record):    
	starting_piece_of_record = record.read_idx(0)
	header_in_starting_piece_of_record, _ = mx.recordio.unpack(starting_piece_of_record)

	flag_indicating_sample_storing = 0
	flag_indicating_record_length = 2

	if header_in_starting_piece_of_record.flag == flag_indicating_sample_storing:
		keys_of_samples = record.keys
	elif header_in_starting_piece_of_record.flag == flag_indicating_record_length:
		keys_of_samples = range(1, int(header_in_starting_piece_of_record.label[0]))
	else:
		pass

	category = {}
	for k in keys_of_samples:
		s = record.read_idx(k)
		header, _ = mx.recordio.unpack(s)
		if header.label in category:
			category[header.label].append(k)
		else:
			category[header.label] = [k]

	return category

transforms = torch.nn.Sequential(
    T.RandomHorizontalFlip(p=0.3),
    T.ConvertImageDtype(torch.float),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
)

def idx_to_data(record, idx, resize = (112, 112)):
	s = record.read_idx(idx)
	header, img = mx.recordio.unpack_img(s, iscolor = 1)
	sample = img.transpose(2, 0, 1)
	sample = transforms(torch.from_numpy(sample).flip(0))    # flip to change BGR to RGB
	if resize != (112, 112):
		sample = T.Resize(resize)(sample)
	return sample, torch.tensor(header.label, dtype=torch.long)


import random
from itertools import chain

class MXFaceDataset(torch.utils.data.Dataset):
	def __init__(self, source, resize = (112, 112)):
		super(MXFaceDataset, self).__init__()
		self.record = mx.recordio.MXIndexedRecordIO(os.path.join(source, 'train.idx'),
													os.path.join(source, 'train.rec'),
													'r')

		self.persons = get_person_id_category(self.record)
		self.idx_to_data = lambda record, idx: idx_to_data(record, idx, resize)


class MXFaceDatasetConventional(MXFaceDataset):
	def __init__(self, source, resize = (112, 112)):
		super(MXFaceDatasetConventional, self).__init__(source, resize)
		self.sample_idx = list(chain(*self.persons.values()))

	def __len__(self):
		return len(self.sample_idx)

	def __getitem__(self, index):
		sample, label = self.idx_to_data(self.record, self.sample_idx[index])
		return {'images':sample,
				'person_ids':label
		}


class MXFaceDatasetBalancedIntraInterClusters(MXFaceDataset):
	def __init__(self, source, resize = (112, 112)):
		super(MXFaceDatasetBalancedIntraInterClusters, self).__init__(source, resize)
		# random.shuffle(self.persons)
		persons_list = list(self.persons.values())
		self.upper = list(chain(*persons_list[::2]))
		self.lower = list(chain(*persons_list[1::2]))

	def __len__(self):
		return int(1e6)

	def __getitem__(self, index):
		same = random.random() > 0.5
		if same:
			pair = random.sample(random.choice(self.persons), 2)
		else:
			pair = (random.choice(self.upper), random.choice(self.lower))

		sample, label = zip(*[self.idx_to_data(self.record, idx) for idx in pair])

		sample = torch.stack(sample)
		label = torch.stack(label)        
		return {'images':sample,
				'person_ids':label
		}

def collate_paired_data(batch):
	batch = {k:torch.cat([b[k] for b in batch], dim = 0) for k in batch[0]}
	return batch

import pickle as pkl

class MXFaceDatasetFromBin(torch.utils.data.Dataset):
	def __init__(self, source, dset, resize = (112, 112)):
		with open(os.path.join(source, dset + '.bin'), 'rb') as f:
			bins, self.issame_list = pkl.load(f, encoding='bytes')

		self.A = []
		self.B = []
		for a, b in zip(bins[0::2], bins[1::2]):
			self.A.append(torch.from_numpy(mx.image.imdecode(a).asnumpy().transpose(2, 0, 1)))
			self.B.append(torch.from_numpy(mx.image.imdecode(b).asnumpy().transpose(2, 0, 1)))
		self.resize = resize

	def __len__(self):
		return len(self.A)

	def __getitem__(self, index):
		a = self.A[index]
		b = self.B[index]
		if self.resize != (112, 112):
			a = T.Resize(resize)(a)
			b = T.Resize(resize)(b)
		return {'id':index,
				'A':a / 255,
				'B':b / 255,
				'same':self.issame_list[index]
				}
