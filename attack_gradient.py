from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import pl_models

import sys
# sys.path.append('Adversarial_Patch_Attack')
sys.path.append('eameo-faceswap-generator')

import faceBlendCommon as fbc
import dlib

import torch
import torch.nn as nn
import torch.nn.functional as F


import numpy as np
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



class Untargeted(nn.Module):
    def __init__(self, model):
        super(Untargeted, self).__init__()
        for name, param in model.named_parameters():  
            param.requires_grad = False
        self.model = model

    def forward(self, perturbated_images, images):
        feat = self.model.extract_feature(torch.cat([perturbated_images, images], dim = 0))
        cos = F.cosine_similarity(feat[:len(feat) //2], feat[len(feat) //2:])
        angles = torch.acos(cos)
        return np.pi - angles

class Targeted(Untargeted):
    def __init__(self, model):
        super(Targeted, self).__init__()

    def forward(self, perturbated_images, images):
        feat = self.model.extract_feature(torch.cat([perturbated_images, images], dim = 0))
        cos = F.cosine_similarity(feat[:len(feat) //2], feat[len(feat) //2:])
        angles = torch.acos(cos)
        return angles


class GlassesAttacker(LightningModule):
    def __init__(self, source = 'thickglasses.png', differentiable_function = None, threshold = None, max_iteration=100):
        super(GlassesAttacker, self).__init__()
        g = np.array(Image.open(source).resize((66, 33)))[:,:,1] > 0
        self.g = torch.from_numpy(g)

        self.v = nn.Parameter(torch.zeros(g.sum() * 3))

        self.differentiable_function = differentiable_function
        self.threshold = threshold

    def get_glasses_mask_for_image(self, img):
        try:
            nose_center = np.flip(get_points(img.squeeze(0))[27], 0)
        except:
            nose_center = np.array([21,55])
        min_point = nose_center - np.array([12,33])
        max_point = nose_center + np.array([21,33])

        m = nn.ZeroPad2d((min_point[1], 112-max_point[1], min_point[0], 112-max_point[0]))
        return m(self.g)

    def forward(self, images, labels = None):#wear
        images.requires_grad = True
        perturbated_images = images.masked_scatter(self.get_glasses_mask_for_image(images).to(self.v.device), torch.tanh(self.v))
        return perturbated_images

    def training_step(self, batch, batch_idx):
        perturbated_images = self(batch['A'])
        images = batch['A']
        self.differentiable_function.eval()
        logits = self.differentiable_function(perturbated_images, images)
        loss = F.threshold(logits, self.threshold, 0.)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

def main(model):        

    source = 'faces_webface_112x112'

    from facedataset import MXFaceDatasetTwin, MXFaceDatasetBalancedIntraInterClusters, collate_paired_data, MXFaceDatasetFromBin

    valid_set_ = MXFaceDatasetFromBin(source, 'lfw')
    valid_set = torch.utils.data.DataLoader(valid_set_, batch_size = 1, shuffle = False, num_workers = 2)

    attacker = GlassesAttacker(differentiable_function = Untargeted(model), threshold = np.pi - 1.3)

    trainer = Trainer(accelerator='gpu' if torch.cuda.is_available() else 'cpu',
              gpus=[opt.device_id],
              max_epochs=opt.max_epochs,
              gradient_clip_val=5,
              callbacks=[ModelCheckpoint(save_last=True)],
             )
    trainer.fit(attacker, valid_set, valid_set, ckpt_path=opt.ckpt if len(opt.ckpt) > 5 else None)
    trainer.test(attacker, valid_set)#, ckpt_path = '/home/ruihan/facereco/lightning_logs/version_13/checkpoints/epoch=3-step=40887.ckpt')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="face models")
    parser.add_argument("--select", type=str, default="arcface", help='which model to train')
    parser.add_argument("--max_epochs", type=int, default=1000, help="max epochs in training")
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



    main(model_under_attack)