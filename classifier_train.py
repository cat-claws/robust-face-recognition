from sklearn.metrics import confusion_matrix, accuracy_score

import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
import numpy as np


from tqdm import tqdm

from pytorch_lightning import Trainer
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint




import argparse




class TwinResNet(LightningModule):
    def __init__(self):
        super(TwinResNet, self).__init__()
        new_in_channels = 6
        model = models.resnet18(pretrained=True)

        layer = model.conv1
                
        # Creating new Conv2d layer
        new_layer = nn.Conv2d(in_channels=new_in_channels, 
                        out_channels=layer.out_channels, 
                        kernel_size=layer.kernel_size, 
                        stride=layer.stride, 
                        padding=layer.padding,
                        bias=layer.bias)

        copy_weights = 0 # Here will initialize the weights from new channel with the red channel weights

        # Copying the weights from the old to the new layer
        new_layer.weight.data[:, :layer.in_channels, :, :] = layer.weight.clone()

        #Copying the weights of the `copy_weights` channel of the old layer to the extra channels of the new layer
        for i in range(new_in_channels - layer.in_channels):
            channel = layer.in_channels + i
            new_layer.weight.data[:, channel:channel+1, :, :] = layer.weight[:, copy_weights:copy_weights+1, : :].clone()
        new_layer.weight = nn.Parameter(new_layer.weight)

        model.conv1 = new_layer
        self.resnet_model = model
        
    def forward(self, images):
        return self.resnet_model(images)

    def training_step(self, batch, batch_idx):
        logits = self(batch['images'])
        loss = F.cross_entropy(logits, batch['same'].long())
        return loss

    def validation_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx, "test")

    def _shared_eval(self, batch, batch_idx, prefix):
        with torch.no_grad():
            logits = self(torch.cat((batch['A'], batch['B']), dim = 1))
            loss = F.cross_entropy(logits, batch['same'].long())
            acc = -accuracy_score(logits.argmax(1).cpu().numpy(), batch['same'].cpu().numpy())

        self.log(f"{prefix}_acc", acc)
        self.log(f"{prefix}_loss", loss)
    
    def configure_optimizers(self):
        # return torch.optim.SGD(self.parameters(), lr=1e-1, momentum=0.9, weight_decay=5e-4)
        return torch.optim.Adam(self.parameters(), lr=1e-4)


def main():
    from facedataset import MXFaceDatasetTwin, MXFaceDatasetFromBin
    
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # train_set_ = MXFaceDatasetConventional(source)
    # train_set = torch.utils.data.DataLoader(train_set_, batch_size = 512, shuffle = True, num_workers = 2, collate_fn=None)

    train_set_ = MXFaceDatasetTwin(source)
    train_set = torch.utils.data.DataLoader(train_set_, batch_size = 256, shuffle = True, num_workers = 2)#, collate_fn=collate_paired_data)

    valid_set_ = MXFaceDatasetFromBin(source, 'agedb_30')
    valid_set = torch.utils.data.DataLoader(valid_set_, batch_size = 256, shuffle = False)

    model = TwinResNet()
    # model.load_state_dict(torch.load(os.path.join(source, 'classifier.pt')))

    trainer = Trainer(accelerator = 'gpu' if torch.cuda.is_available() else 'cpu',
                    gpus = [opt.device_id],
                    max_epochs = 15,
                    gradient_clip_val = 5,
                    callbacks = [ModelCheckpoint(monitor='val_loss')]
                    )
    trainer.fit(model, train_set, valid_set)
    trainer.test(model, valid_set, ckpt_path = 'best')

    torch.save(model.state_dict(), os.path.join(source, 'classifier.pt'))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="face models")
    parser.add_argument("--source", type=str, default="faces_webface_112x112", help='path of files to process')
    parser.add_argument("--device_id", type=int, default=1, help="GPU id")

    opt = parser.parse_args()

    source = opt.source

    main()
