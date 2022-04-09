import pl_models

import sys
sys.path.append('eameo-faceswap-generator')

import faceBlendCommon as fbc
import dlib

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import numpy as np
from tqdm import tqdm
from PIL import Image


import argparse


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("eameo-faceswap-generator/shape_predictor_68_face_landmarks.dat")
get_points = lambda x: np.array(fbc.getLandmarks(detector, predictor, ((x.numpy()/2+0.5) * 255).astype('uint8').transpose(1, 2, 0)))

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





def get_angle(model, perturbated_image, target, same):
    model.eval()
    feat = model.extract_feature(torch.cat([perturbated_image, target], dim = 0).to(next(model.parameters()).device))
    cos = F.cosine_similarity(feat[:len(feat) //2], feat[len(feat) //2:])
    angles = torch.acos(cos)
    return (np.pi - angles) if same else angles 

def get_glasses_mask(source):
    g = np.array(Image.open(source).resize((66, 33)))[:,:,1] > 0
    g = torch.from_numpy(g)
    return g

def glasses_initialization(g):
    v = torch.zeros(g.sum() * 3)
    return v

def get_glasses_mask_for_image(img, g):
    try:
        nose_center = np.flip(get_points(img.squeeze(0))[27], 0)
    except:
        nose_center = np.array([21,55])
    min_point = nose_center - np.array([12,33])
    max_point = nose_center + np.array([21,33])

    m = nn.ZeroPad2d((min_point[1], 112-max_point[1], min_point[0], 112-max_point[0]))
    return m(g)

def wear(img, glasses_vector, g):
    return img.masked_scatter(get_glasses_mask_for_image(img, g), glasses_vector)



def patch_attack(model, image, glasses_parameters, glasses_mask, target, same, angle_threshold, lr=1, max_iteration=100):

    target_angle, count = np.pi, 0
    perturbated_image = wear(image, glasses_parameters, glasses_mask)
    
    angle_threshold = (np.pi - angle_threshold) if same else angle_threshold

    while target_angle > (angle_threshold - 0) and count < max_iteration:
        # Optimize the patch
        perturbated_image = (np.rint((perturbated_image / 2 + 0.5) *  255) / 255 - 0.5) * 2
        perturbated_image.requires_grad = True

        angle = get_angle(model, perturbated_image, target, same)

        target_angle = angle.item()
        if count %100 == 0:
            print(target_angle)
            
        count += 1
        angle.backward()

        patch_grad = perturbated_image.grad.clone().cpu().masked_select(get_glasses_mask_for_image(image, glasses_mask))
        perturbated_image.grad.data.zero_()

        glasses_parameters = - lr * patch_grad + glasses_parameters.type(torch.FloatTensor)
        glasses_parameters = torch.clamp(glasses_parameters, min=-1, max=1)
        
        # Test the patch
        perturbated_image = wear(image, glasses_parameters, glasses_mask)

        # Record the statistics
        with open(opt.select + '.txt', 'a') as f:
            f.write(str(target_angle) + '\n')

    perturbated_image = perturbated_image.cpu().numpy()
    glasses_parameters = glasses_parameters.cpu()
    return perturbated_image, glasses_parameters, target_angle


def main(model, folder):
        

    from facedataset import MXFaceDatasetTwin, MXFaceDatasetBalancedIntraInterClusters, collate_paired_data, MXFaceDatasetFromBin

    valid_set_ = MXFaceDatasetFromBin(opt.source, 'lfw')
    valid_set = torch.utils.data.DataLoader(valid_set_, batch_size = 1, shuffle = False, num_workers = 2)

    for name, param in model.named_parameters():  
        param.requires_grad = False

    glasses_mask =  get_glasses_mask('thickglasses.png')
    glasses_parameters = glasses_initialization(glasses_mask)
    print('The shape of the patch is', glasses_parameters.shape)

    best_patch_success_rate = 0
    best_patch_success_rate = 0
    for epoch in range(1):
        train_total, train_actual_total, train_success = 0, 0, 0
        for batch in tqdm(valid_set):
            assert batch['same'].shape[0] == 1, 'Only one picture should be loaded each time.'
            batch_id =  batch['id'].item()
            same = batch['same'].item()
            
            if not same:
                continue

            if os.path.exists(f'{folder}/{batch_id}_perturbed.npy'):
                continue

            with open(opt.select + '.txt', 'a') as f:
                f.write('id' + str(batch_id) + '\n')
                f.write(str(same) + '\n')

            train_total += batch['A'].shape[0]
            image = batch['A']
            target = batch['B']

            angle = get_angle(model, image, target, same)

            train_actual_total += 1

            perturbated_image, glasses_parameters, target_angle = patch_attack(model, image, glasses_parameters, glasses_mask, target, same, 1.3, 1., 20000)
            print(f'original {angle.item():.3f}, optimized angle: {target_angle:.3f}')

            if not os.path.isdir(folder):
                os.mkdir(folder)
            np.save(f'{folder}/{batch_id}_glasses_parameters', glasses_parameters.numpy())
            np.save(f'{folder}/{batch_id}_perturbed', np.transpose(perturbated_image.squeeze(0), (1, 2, 0)) / 2 + 0.5)

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="face models")
    parser.add_argument("--select", type=str, default="arcface", help='which model to train')
    parser.add_argument("--max_epochs", type=int, default=100, help="max epochs in training")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size in training")
    parser.add_argument("--ckpt", type=str, default="", help='pytorch lightning checkpoint')
    parser.add_argument("--state_dict", type=str, default="arcface18.pth", help='pytorch load_state_dict checkpoint')
    parser.add_argument("--source", type=str, default="faces_webface_112x112", help='directory of files to process')
    parser.add_argument("--valid_source", type=str, default="lfw", help='path of validation files')
    parser.add_argument("--num_persons", type=int, default=13938, help="number of persons in training set")
    parser.add_argument("--embedding_dim", type=int, default=512, help="embedding dimension")
    parser.add_argument("--structure", type=str, default="resnet18", help='resnet models')
    parser.add_argument('--embedding_dims', help='delimited list input for embeddings', type=str, default = "102,102,102,102,104")
    parser.add_argument('--structures', help='delimited list input for structures', type=str, default="resnet10,resnet10,resnet10,resnet10,resnet10")
    parser.add_argument("--device_id", type=int, default=1, help="GPU id")

    opt = parser.parse_args()

    if opt.select == 'arcface':
        model_under_attack = pl_models.ArcFace(out_features = opt.num_persons, embeddings = opt.embedding_dim, structure = opt.structure)
    elif opt.select == 'arcorgans':
        model_under_attack = pl_models.ArcOrgans(out_features = opt.num_persons, embeddings = opt.embedding_dim, structure = opt.structure)
    elif opt.select == 'arcsegface':
        embedding_dims = [int(x) for x in opt.embedding_dims.split(',')]
        structures = opt.structures.split(',')
        model_under_attack = pl_models.ArcSegFace(out_features = opt.num_persons, embeddings = embedding_dims, structures = structures)

    try:
        model_under_attack.load_state_dict(torch.load(opt.state_dict))
    except:
        model_under_attack.backbone.load_state_dict(torch.load(opt.state_dict))

    model_under_attack.eval();

    device = f'cuda:{opt.device_id}' if torch.cuda.is_available() else 'cpu'

    main(model_under_attack, 'atk_glass_' + opt.select)