import sys
sys.path.append('face-parsing.PyTorch')

from model import BiSeNet

import torch

import os
import numpy as np
from PIL import Image

import torchvision.transforms as transforms
import cv2

device = 'cuda' if torch.cuda.is_available() else 'cpu'


n_classes = 19
net = BiSeNet(n_classes=n_classes)
net.to(device)
net.load_state_dict(torch.load('79999_iter.pth', map_location = device))
net.eval()

to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

def segment(input_img):

    with torch.no_grad():
        img = Image.fromarray(np.uint8(input_img))
        image = img.resize((512, 512), Image.BILINEAR)
        img = to_tensor(image)
        img = torch.unsqueeze(img, 0)
        img = img.to(device)

        out = net(img)[0]
        mask = out.squeeze(0).cpu().argmax(0).unsqueeze(0)
        mask = transforms.Resize(input_img.shape[:2], interpolation = transforms.InterpolationMode.NEAREST)(mask).numpy().squeeze(0)


    return mask

def binary_mask(input_img):
    mask = segment(input_img)
    return np.logical_or(np.logical_and(mask > 0, mask < 7),np.logical_and(mask > 9, mask < 14))

def remove_background(input_img):
    input_img[binary_mask(input_img) == 0] = 0
    return input_img




import os
import mxnet as mx

source = 'faces_webface_112x112'
output = 'faces_webface_112x112_fine'

valid_dataset = 'lfw'

imgrec = mx.recordio.MXIndexedRecordIO(os.path.join(source, 'train.idx'), os.path.join(source, 'train.rec'), 'r')

import numpy as np

s = imgrec.read_idx(0)
header, _ = mx.recordio.unpack(s)
if header.flag > 0:
    header0 = (int(header.label[0]), int(header.label[1]))
    imgidx = np.array(range(1, int(header.label[0])))
else:
    imgidx = np.array(list(imgrec.keys))

from tqdm import tqdm

record = mx.recordio.MXIndexedRecordIO(os.path.join(output, 'train.idx'), os.path.join(output, 'train.rec'), 'w')
s = imgrec.read_idx(0)
record.write_idx(0, s)

for i in tqdm(imgidx):
    s = imgrec.read_idx(i)
    header, img = mx.recordio.unpack(s)
    sample = mx.image.imdecode(img).asnumpy()
    try:
        # sample = organ_image(sample, get_points(sample))
        sample = np.flip(remove_background(sample), 2)
        packed_s = mx.recordio.pack_img(header, sample)
        record.write_idx(i, packed_s)
    except:
        pass
record.close()


import pickle as pkl


with open(os.path.join(source, valid_dataset + '.bin'), 'rb') as f:
    bins, issame_list = pkl.load(f, encoding='bytes')

A = []
B = []
S = []
for s, a, b in tqdm(zip(issame_list, bins[0::2], bins[1::2])):
    try:
        a = mx.image.imdecode(a).asnumpy()
        a = remove_background(a)

        b = mx.image.imdecode(b).asnumpy()
        b = remove_background(b)

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
    packed_i = mx.recordio.pack_img(header, x)
    _, img_only = mx.recordio.unpack(packed_i)
    bins[i] = img_only

with open(os.path.join(output, valid_dataset + '.bin'), 'wb') as f:
    pkl.dump((bins, S), f)