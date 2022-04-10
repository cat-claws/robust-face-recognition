import torch
import sys

sys.path.append('robust-face-recognition')
import pl_models
model = pl_models.ArcFace(structure = 'r18')
model.backbone.load_state_dict(torch.load('arcface18.pth'))



import os
import numpy as np
from PIL import Image

import torchvision.transforms as transforms
import cv2

device = 'cuda' if torch.cuda.is_available() else 'cpu'

from attack_gradient_global import GlassesAttacker, Untargeted
attacker = GlassesAttacker(differentiable_function = Untargeted(model), threshold = np.pi - 1.324)
ckpt = torch.load('/home/ruihan/facereco/attack_logs/arcface/version_6/checkpoints/last.ckpt')
attacker.load_state_dict(ckpt['state_dict'])

def wear_glasses(input_img):
    input_img = ((input_img / 255 - 0.5) * 2).transpose(2, 0, 1)
    perturbed = attacker(torch.from_numpy(input_img).unsqueeze(0).float())
    # input_img[binary_mask(input_img) == 0] = 0
    perturbed = ((perturbed.detach().numpy()/2+0.5) * 255).astype('uint8').squeeze(0).transpose(1, 2, 0)
    return perturbed




import os
import mxnet as mx

source = 'faces_webface_112x112'
output = 'faces_webface_112x112_atk_glass_arcface18'

valid_dataset = 'lfw'

# imgrec = mx.recordio.MXIndexedRecordIO(os.path.join(source, 'train.idx'), os.path.join(source, 'train.rec'), 'r')

# import numpy as np

# s = imgrec.read_idx(0)
# header, _ = mx.recordio.unpack(s)
# if header.flag > 0:
#     header0 = (int(header.label[0]), int(header.label[1]))
#     imgidx = np.array(range(1, int(header.label[0])))
# else:
#     imgidx = np.array(list(imgrec.keys))

from tqdm import tqdm

# record = mx.recordio.MXIndexedRecordIO(os.path.join(output, 'train.idx'), os.path.join(output, 'train.rec'), 'w')
# s = imgrec.read_idx(0)
# record.write_idx(0, s)

# for i in tqdm(imgidx):
#     s = imgrec.read_idx(i)
#     header, img = mx.recordio.unpack(s)
#     sample = mx.image.imdecode(img).asnumpy()
#     try:
#         # sample = organ_image(sample, get_points(sample))
#         sample = np.flip(wear_glasses(sample), 2)
#         packed_s = mx.recordio.pack_img(header, sample)
#         record.write_idx(i, packed_s)
#     except:
#         pass
# record.close()


import pickle as pkl


with open(os.path.join(source, valid_dataset + '.bin'), 'rb') as f:
    bins, issame_list = pkl.load(f, encoding='bytes')

A = []
B = []
S = []
for s, a, b in tqdm(zip(issame_list, bins[0::2], bins[1::2])):
    # try:
    a = mx.image.imdecode(a).asnumpy()
    a = wear_glasses(a)

    b = mx.image.imdecode(b).asnumpy()
    # b = wear_glasses(b)

    A.append(a)
    B.append(b)
    S.append(s)
    # except:
    #     pass

bins = []
header = mx.recordio.IRHeader(0, 5, 7, 0)
for i in range(len(A)):
    bins.append(np.flip(A[i], 2))
    bins.append(np.flip(B[i], 2))
for i, x in enumerate(bins):
    packed_i = mx.recordio.pack_img(header, x, quality=9, img_fmt='.png')
    _, img_only = mx.recordio.unpack(packed_i)
    bins[i] = img_only

with open(os.path.join(output, valid_dataset + '.bin'), 'wb') as f:
    pkl.dump((bins, S), f)