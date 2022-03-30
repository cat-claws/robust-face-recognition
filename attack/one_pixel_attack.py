from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, accuracy_score


import torch
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

import os
import sys
import numpy as np

sys.path.append('one-pixel-attack-pytorch')
sys.path.append('insightface/recognition/arcface_torch')

from backbones import get_model

model = get_model('r18')
model.load_state_dict(torch.load('arcface18.pth', map_location = device))
model = model.to(device)
model.eval();


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The ResNet has {count_parameters(model):,} trainable parameters')

import torch
import torch.nn as nn

import torch.nn.functional as F
import torchvision.transforms as T

from differential_evolution import differential_evolution


num_persons = 13938
source = 'faces_webface_112x112'

from facedataset import MXFaceDatasetBalancedIntraInterClusters, collate_paired_data, MXFaceDatasetFromBin

# train_set_ = MXFaceDatasetPair(source)
# train_set = torch.utils.data.DataLoader(train_set_, batch_size = 64, shuffle = True, num_workers = 2, collate_fn=None)

# train_set_ = MXFaceDatasetBalancedIntraInterClusters(source, resize = 64)
# train_set = torch.utils.data.DataLoader(train_set_, batch_size = 32, shuffle = True, num_workers = 2, collate_fn=collate_paired_data)

valid_set_ = MXFaceDatasetFromBin(source, 'lfw')
valid_set = torch.utils.data.DataLoader(valid_set_, batch_size = 1, shuffle = False, num_workers = 1)





tranfrom = T.Compose([
                      T.ToTensor(),
                      T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                      ])

def perturb_image(xs, img):
	if xs.ndim < 2:
		xs = np.array([xs])
	batch = len(xs)
	imgs = img.repeat(batch, 1, 1, 1)
	xs = xs.astype(int)

	count = 0
	for x in xs:
		pixels = np.split(x, len(x)/5)
		
		for pixel in pixels:
			x_pos, y_pos, r, g, b = pixel
			imgs[count, 0, x_pos, y_pos] = (r/255.0-0.4914)/0.2023
			imgs[count, 1, x_pos, y_pos] = (g/255.0-0.4822)/0.1994
			imgs[count, 2, x_pos, y_pos] = (b/255.0-0.4465)/0.2010
		count += 1

	return imgs

def predict_distances(xs, A, B, net):
    A_ = perturb_image(xs, A.clone())
    with torch.no_grad():
        feat = net(torch.cat((A_, B.unsqueeze(0)), dim = 0).to(device))
        feat = feat.cpu().numpy()
        feat_A_ = feat[:-1]
        feat_B = feat[-1:]
        distances = np.linalg.norm(normalize(feat_A_) - normalize(feat_B), axis=1)
    return distances

def attack_success(x, A, B, net, same, verbose=False):
    distance = predict_distances(x, A, B, net)
    if verbose:
        print(f'Distance: {distance}')
        
    if (not same and distance < 1.207) or (same and distance > 1.207):
        return True

def attack(A, B, net, same, pixels=1, maxiter=75, popsize=400, verbose=False):
	# A: 1*3*W*H tensor
    # B: 1*3*W*H tensor

    bounds = [(0,112), (0,112), (0,255), (0,255), (0,255)] * pixels

    popmul = max(1, popsize // len(bounds))

    predict_fn = lambda xs: predict_distances(xs, A, B, net) * (int(same) * (-2) + 1)
    callback_fn = lambda x, convergence: attack_success(x, A, B, net, same, verbose)

    inits = np.zeros([popmul*len(bounds), len(bounds)])
    for init in inits:
        for i in range(pixels):
            init[i*5+0] = np.random.random()*112
            init[i*5+1] = np.random.random()*112
            init[i*5+2] = np.random.normal(128,127)
            init[i*5+3] = np.random.normal(128,127)
            init[i*5+4] = np.random.normal(128,127)

    attack_result = differential_evolution(predict_fn, bounds, maxiter=maxiter, popsize=popmul,
        recombination=1, atol=-1, callback=callback_fn, polish=False, init=inits)

    predicted_dists = predict_distances(attack_result.x, A, B, net)

    return predicted_dists, attack_result.x.astype(int)


from tqdm import tqdm

def get_image_distance(A, B, net):
	with torch.no_grad():
		feat = net(torch.stack((A, B)).to(device))
		feat = normalize(feat.cpu().numpy())
	return np.linalg.norm(feat[0] - feat[1], axis=0)

import json

with open('lfw_arcface_distances.json', 'r') as f:
    dists_ = json.loads(f.read())

dists = {}
for k, v in dists_.items():
    if int(k) % 600 < 300:
        if v > 0.95:
            dists[int(k)] = v

def attack_all(net, data, pixels=1, maxiter=75, popsize=400, verbose=False):

	attacks = {}
	distances = {}

	for k, sample in enumerate(tqdm(data)):
		if k not in dists:
			continue

		same = sample['same'].item()#.cpu().numpy()
			
		dist, x = attack(sample['A'].squeeze(0), sample['B'].squeeze(0), net, same, pixels=pixels, maxiter=maxiter, popsize=popsize, verbose=verbose)

		print("[(x,y) = (%d,%d) and (R,G,B)=(%d,%d,%d)]"%(
			x[0],x[1],x[2],x[3],x[4]))
		
		attacks[sample['id'].item()] = x
		distances[sample['id'].item()] = dist


	return attacks, distances

results = attack_all(model,
                      valid_set,
                      pixels=5,
                      maxiter=400,
                      popsize=400,
                      verbose=False
                      )

import pickle
with open('attacks.pkl', 'wb') as f:
	pickle.dump(results, f)