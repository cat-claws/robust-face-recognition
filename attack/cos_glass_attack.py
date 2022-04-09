import sys
sys.path.append('Adversarial_Patch_Attack')
sys.path.append('eameo-faceswap-generator')
from PIL import Image

import faceBlendCommon as fbc
import cv2
# Import modules
import dlib

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision import models

import os
import numpy as np
from tqdm import tqdm
from PIL import Image

from arcface_train import ArcFace
import json

import argparse

# device = 'cuda' if torch.cuda.is_available() else 'cpu'


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
    feat = model.extract_feature(torch.cat([perturbated_image, target], dim = 0).to(next(model.parameters()).device))
    cos = F.cosine_similarity(feat[:len(feat) //2], feat[len(feat) //2:])
    angles = torch.acos(cos)
    return np.pi - angles if same else angles 

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
    model.eval()
    # glasses_parameters = torch.from_numpy(glasses_parameters)
    # mask = torch.from_numpy(mask)
    target_angle, count = (np.pi - int(same) * np.pi), 0
    perturbated_image = wear(image, glasses_parameters, glasses_mask)
    
    angle_threshold = np.pi - angle if same else angle_threshold

    while target_angle > (angle_threshold - 0) and count < max_iteration:
        # Optimize the patch
        # perturbated_image = perturbated_image.to(device)
        perturbated_image = (np.rint((perturbated_image / 2 + 0.5) * 255) / 255 - 0.5) * 2
        perturbated_image.requires_grad = True

        # dist = get_distance(backbone, perturbated_image, target, same)
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
        # perturbated_image = torch.clamp(perturbated_image, min=-3, max=3)
        # perturbated_image = perturbated_image.to(device)

        # output = model(perturbated_image)

    perturbated_image = perturbated_image.cpu().numpy()
    glasses_parameters = glasses_parameters.cpu()#.numpy()
    return perturbated_image, glasses_parameters, target_angle


def main():
        

    source = 'faces_webface_112x112'

    from facedataset import MXFaceDatasetTwin, MXFaceDatasetBalancedIntraInterClusters, collate_paired_data, MXFaceDatasetFromBin

    valid_set_ = MXFaceDatasetFromBin(source, 'lfw')
    valid_set = torch.utils.data.DataLoader(valid_set_, batch_size = 1, shuffle = False, num_workers = 2)

    # model = ArcFace(out_features = 13938, structure="r18").to(device)
    model = ArcSegFace()
    # model.backbone.load_state_dict(torch.load('arcface18.pth', map_location=device))
    # model.load_state_dict(torch.load('faces_webface_112x112_no_organs/arcface.pt', map_location=device))
    model.load_state_dict(torch.load('faces_webface_112x112/arcorgans_fork.pt', map_location=device))
    model.eval();

    for name, param in model.named_parameters():  
        param.requires_grad = False
    # with open('lfw_arcface_distances.json', 'r') as f:
    #     unattacked_dists_ = json.loads(f.read())
    #     unattacked_dists = {}
    #     for k, v in unattacked_dists_.items():
    #         unattacked_dists[int(k)] = v

    # patch = patch_initialization(image_size=(3, 112, 112), noise_percentage=0.1)
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
            if not same:#) or (batch_id < 9):# in [68, 69, 112, 185]:
                continue
            # if unattacked_dists[batch_id] > 1.45:
            #     continue
            # if os.path.exists(f'adv_glasses_on_organs/{batch_id}_perturbed.npy'):
            #     continue
            train_total += batch['A'].shape[0]
            image = batch['A']#.to(device)
            target = batch['B']#.to(device)

            angle = get_angle(model, image, target, same)

            # if predicted[0] != label and predicted[0].data.cpu().numpy() != args.target:
            train_actual_total += 1
            # glasses_parameters, mask, x_location, y_location = mask_generation(patch=patch, image_size=(3, 112, 112))

            try:
                perturbated_image, glasses_parameters, target_angle = patch_attack(model, image, glasses_parameters, glasses_mask, target, same, 1.3, 1., 20000)
                print(f'original {angle:.3f}, optimized angle: {target_angle:.3f}')
            except:
                print(f'number {batch_id} fails to wear glasses.')
                continue


            # if target_angle < 1.3:
            #     train_success += 1
            # print(patch.shape)
            # patch = glasses_parameters[0][:, x_location:x_location + patch.shape[1], y_location:y_location + patch.shape[2]]

            # mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            np.save(f'adv_glasses/{batch_id}_glasses_parameters', glasses_parameters.numpy())# * std + mean)
            np.save(f'adv_glasses/{batch_id}_perturbed', np.transpose(perturbated_image.squeeze(0), (1, 2, 0)) / 2 + 0.5)
            # plt.imshow(np.clip(np.transpose(patch, (1, 2, 0)) * std + mean, 0, 1))
            # plt.savefig("adv_glasses_on_organs/" + str(epoch) + " patch.png")
            # print("Epoch:{} Patch attack success rate on trainset: {:.3f}%".format(epoch, 100 * train_success / train_actual_total))
        # train_success_rate = test_patch(args.patch_type, args.target, patch, test_loader, model)
        # print("Epoch:{} Patch attack success rate on trainset: {:.3f}%".format(epoch, 100 * train_success_rate))
        # test_success_rate = test_patch(args.patch_type, args.target, patch, test_loader, model)
        # print("Epoch:{} Patch attack success rate on testset: {:.3f}%".format(epoch, 100 * test_success_rate))

        # Record the statistics
        # with open(args.log_dir, 'a') as f:
        #     writer = csv.writer(f)
        #     writer.writerow([epoch, train_success_rate, test_success_rate])
        # test_success_rate = train_success
        # print(test_success_rate)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="face models")
    # parser.add_argument("--max_epochs", type=int, default=100, help="max epochs in training")
    # parser.add_argument("--structure", type=str, default="resnet18", help='resnet models')
    # parser.add_argument("--ckpt", type=str, default="", help='pytorch lightning checkp')
    # parser.add_argument("--source", type=str, default="faces_webface_112x112", help='path of files to process')
    # parser.add_argument("--num_persons", type=int, default=13938, help="number of persons in training set")
    # parser.add_argument("--embedding_dim", type=int, default=512, help="embedding dimension")
    parser.add_argument("--device_id", type=int, default=1, help="GPU id")

    opt = parser.parse_args()

    # num_persons = opt.num_persons
    # source = opt.source
    device = f'cuda:{opt.device_id}' if torch.cuda.is_available() else 'cpu'


    main()