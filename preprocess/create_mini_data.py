# Import modules
import sys, dlib
sys.path.append('eameo-faceswap-generator')

import numpy as np
import faceBlendCommon as fbc
# import matplotlib.pyplot as plt
from PIL import Image

# get landmark
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("eameo-faceswap-generator/shape_predictor_68_face_landmarks.dat")
get_points = lambda x: np.array(fbc.getLandmarks(detector, predictor, x))

FACIAL_LANDMARKS_IDXS = {
	"mouth": (48, 68),
	# "right_eyebrow": (17, 22),
	# "left_eyebrow": (22, 27),
	# "right_eye": (36, 42),
	# "left_eye": (42, 48),
	"nose": (27, 35),
	# "jaw": (0, 17),
	"eye" : (36, 48),
	"eyebrow" : (17, 27)
 }

def get_crop_boundary(points_subset):
    min_point = np.array([np.min(points_subset[:,0]), np.min(points_subset[:,1])])
    max_point = np.array([np.max(points_subset[:,0]), np.max(points_subset[:,1])])
    return min_point, max_point

def organ_image(image, points, organ = 'nose'):
    start, end = FACIAL_LANDMARKS_IDXS[organ]
    
    min_point, max_point = get_crop_boundary(points[start:end])
    new_image = np.zeros_like(image)
    if organ == "nose":
        new_image[min_point[1]:max_point[1],min_point[0]-5:max_point[0]+5,:] = image[min_point[1]:max_point[1],min_point[0]-5:max_point[0]+5,:]
    else:
        new_image[min_point[1]:max_point[1],min_point[0]:max_point[0],:] = image[min_point[1]:max_point[1],min_point[0]:max_point[0],:]


    return new_image

def organ_only_image(image, points):
    new_image = np.zeros_like(image)
    for organ in FACIAL_LANDMARKS_IDXS:

        start, end = FACIAL_LANDMARKS_IDXS[organ]
        
        min_point, max_point = get_crop_boundary(points[start:end])
        
        new_image[min_point[1]:max_point[1],min_point[0]:max_point[0],:] = image[min_point[1]:max_point[1],min_point[0]:max_point[0],:]

    return new_image

def no_organ_image(image, points):
    new_image = np.array(image, copy=True)
    for organ in FACIAL_LANDMARKS_IDXS:

        start, end = FACIAL_LANDMARKS_IDXS[organ]
        
        min_point, max_point = get_crop_boundary(points[start:end])
        
        new_image[min_point[1]:max_point[1],min_point[0]:max_point[0],:] = np.mean(image)

    return new_image

from shapely import geometry

def get_fan(polygon, shape):
    line = geometry.Polygon(list(polygon))
    contains = lambda x: line.contains(geometry.Point(*x))
    
    grid = np.indices(shape)
    
    fan = np.zeros(shape)

    min_point, max_point = get_crop_boundary(np.array(list(polygon)))

    old_patch = grid[:, min_point[0]:max_point[0],min_point[1]:max_point[1]]
    patch_shape = (old_patch.shape[1], old_patch.shape[2])
    old_patch_list = old_patch.transpose(1, 2, 0).reshape(-1, 2).tolist()
    new_patch = np.array(list(map(contains, old_patch_list))).reshape(patch_shape)        
    fan[min_point[0]:max_point[0],min_point[1]:max_point[1]] = new_patch


    return fan

def get_jaw_fan(sample, points):
    jaw = points[0:17]
    nose = [points[30]] * 16
    fans = map(lambda x: get_fan(x, sample.shape[:-1]), zip(jaw[:-1], jaw[1:], nose))

    fans = np.stack(fans, axis=2)
    fans = np.stack((np.sum(fans[:,:,::3], -1), np.sum(fans[:,:,1::3], -1), np.sum(fans[:,:,2::3], -1)), 2).transpose(1, 0, 2) * 255
    return fans

"""#Prepare images"""

# !pip install mxnet >/dev/null

import os
import mxnet as mx

source = 'faces_webface_112x112_glasses'
# output = 'faces_webface_112x112_organs'
# output = 'faces_webface_112x112_no_organs'
output = 'faces_webface_112x112_glasses_organs'


from tqdm import tqdm

# imgrec = mx.recordio.MXIndexedRecordIO(os.path.join(source, 'train.idx'), os.path.join(source, 'train.rec'), 'r')

# import numpy as np

# s = imgrec.read_idx(0)
# header, _ = mx.recordio.unpack(s)
# if header.flag > 0:
#     header0 = (int(header.label[0]), int(header.label[1]))
#     imgidx = np.array(range(1, int(header.label[0])))
# else:
#     imgidx = np.array(list(imgrec.keys))


# record = mx.recordio.MXIndexedRecordIO(os.path.join(output, 'train.idx'), os.path.join(output, 'train.rec'), 'w')
# s = imgrec.read_idx(0)
# record.write_idx(0, s)

# for i in tqdm(imgidx):
#     s = imgrec.read_idx(i)
#     header, img = mx.recordio.unpack(s)
#     sample = mx.image.imdecode(img).asnumpy()
#     try:
#         sample = organ_image(sample, get_points(sample))
#         packed_s = mx.recordio.pack_img(header, sample)
#         record.write_idx(i, packed_s)
#     except:
#         pass
# record.close()


import pickle as pkl


with open(os.path.join(source, 'lfw.bin'), 'rb') as f:
    bins, issame_list = pkl.load(f, encoding='bytes')

A = []
B = []
S = []
for s, a, b in tqdm(zip(issame_list, bins[0::2], bins[1::2])):
    try:
        a = mx.image.imdecode(a).asnumpy()
        pa = get_points(a)
        a = organ_only_image(a, pa)

        b = mx.image.imdecode(b).asnumpy()
        pb = get_points(b)
        b = organ_only_image(b, pb)

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
# with open(os.path.join(output, 'lfw.pkl'), 'wb') as f:
#     pkl.dump((S, A, B), f)
