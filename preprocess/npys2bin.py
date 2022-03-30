import os
import numpy as np
import pickle as pkl

import mxnet as mx
from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser(description="n.a.")
parser.add_argument("--source", type=str, default="faces_webface_112x112", help='path of files to process')

parser.add_argument("--input_folder", type=str, default="adv_rect", help='path of files to process')
parser.add_argument("--output", type=str, default="faces_webface_112x112_rect", help='path of files to process')
# parser.add_argument("--num_persons", type=int, default=13938, help="number of persons in training set")
# parser.add_argument("--embedding_dim", type=int, default=512, help="embedding dimension")
# parser.add_argument("--device_id", type=int, default=1, help="GPU id")

opt = parser.parse_args()

source = opt.source
output = opt.output

mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

with open(os.path.join(source, 'lfw.bin'), 'rb') as f:
    bins, issame_list = pkl.load(f, encoding='bytes')

A = []
B = []
S = []
for i, (s, a, b) in tqdm(enumerate(zip(issame_list, bins[0::2], bins[1::2]))):
	try:
		# a = mx.image.imdecode(a).asnumpy()
		# pa = get_points(a)
		# a = organ_image(a, pa)
		# a = np.rint(np.clip((((np.load(f'{opt.input_folder}/{i}_perturbed.npy') - mean) / std / 2 + 0.5) * 255), 0, 255)).astype(int)
		a = np.rint(np.load(os.path.join(opt.input_folder, f'{i}_perturbed.npy')) * 255).astype(int)
		# - mean) / std /2 +0.5)*255

		b = mx.image.imdecode(b).asnumpy()
		# pb = get_points(b)
		# b = organ_image(b, pb)

		A.append(a)
		B.append(b)
		S.append(s)
	except:
		pass

bins = []
header = mx.recordio.IRHeader(0, 5, 7, 0)
for i in range(len(A)):
    bins.append(np.flip(A[i], 2))
    bins.append(np.flip(B[i], 2))
for i, x in enumerate(bins):
    packed_i = mx.recordio.pack_img(header, x, quality=9, img_fmt='.png')
    _, img_only = mx.recordio.unpack(packed_i)
    bins[i] = img_only

with open(os.path.join(output, 'lfw.bin'), 'wb') as f:
    pkl.dump((bins, S), f)