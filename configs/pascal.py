# Basic training configuration
from torchinfo import summary
import os
from functools import partial

import albumentations as A
from torchsummary import summary
from torchvision import transforms as T
import torchvision
import torch
import cv2
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
from albumentations.pytorch import ToTensorV2 as ToTensor
from dataflow import get_train_val_loaders, ignore_mask_boundaries
from torchvision.models.segmentation import deeplabv3_resnet101, deeplabv3_resnet50, deeplabv3_mobilenet_v3_large, fcn_resnet50, fcn_resnet101, lraspp_mobilenet_v3_large
from torchvision.models.segmentation import DeepLabV3_MobileNet_V3_Large_Weights, DeepLabV3_ResNet101_Weights
from net import official_deeplab_resnet, pretrained_backbone_tconv_head, tconv_backbone_conv_head, tconv, tconv1, tconv_fcn, tconv_fcn1, tconv_backbone_tconv_head, tconv_test, tconv_test1
# ##############################
# Global configs
# ##############################

seed = 21
device = "cuda"
debug = False
# Use AMP with torch native
with_amp = True


batch_size = 16  # total batch size
val_batch_size = batch_size * 2
num_workers = 12  # total num workers per node
val_interval = 3
# grads accumulation:
accumulation_steps = 4

# 144 not working for new_tconv
#val_img_size = 80
#train_img_size = 80
#val_img_size = 128
#train_img_size = 128
val_img_size = 256
train_img_size = 256

# ##############################
# Setup Dataflow
# ##############################

#print(os.environ)
assert "DATASET_PATH" in os.environ
data_path = os.environ["DATASET_PATH"]

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)


train_transforms = A.Compose(
    [
        A.Resize(val_img_size, val_img_size),
        A.Normalize(mean=mean, std=std),
        ignore_mask_boundaries,
        ToTensor(),
    ]
)

val_transforms = A.Compose(
    [
        A.Resize(val_img_size, val_img_size),
        A.Normalize(mean=mean, std=std),
        ignore_mask_boundaries,
        ToTensor(),
    ]
)


train_loader, val_loader, train_eval_loader = get_train_val_loaders(
    root_path=data_path,
    train_transforms=train_transforms,
    val_transforms=val_transforms,
    batch_size=batch_size,
    num_workers=num_workers,
    val_batch_size=val_batch_size,
    limit_train_num_samples=100 if debug else None,
    limit_val_num_samples=100 if debug else None,
 )
 
# ##############################
# Setup model
# ##############################

num_classes = 21
#model = deeplabv3_resnet101(weights=None, num_classes=num_classes, weights_backbone=None)
#model = deeplabv3_resnet50(num_classes=num_classes, weights_backbone=None)
#model = fcn_resnet101(num_classes=num_classes, weights_backbone=None)
#model = fcn_resnet50(num_classes=num_classes, weights_backbone=None)
#model = deeplabv3_mobilenet_v3_large(num_classes=num_classes, weights_backbone=None)
#model = deeplabv3_resnet101(weights=None, num_classes=num_classes)
#model = deeplabv3_resnet50(num_classes=num_classes)
#model = fcn_resnet50(num_classes=num_classes)
#model = fcn_resnet101(num_classes=num_classes)
#model = deeplabv3_mobilenet_v3_large(num_classes=num_classes)
#model =lraspp_mobilenet_v3_large(num_classes=num_classes)
#model =  torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(weights=DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1, num_classes=num_classes, aux_loss=True)
#model =  torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(num_classes=num_classes, aux_loss=False)
#model = tconv_backbone_conv_head(device, num_classes)
model = tconv_backbone_tconv_head(device, num_classes)
#model = official_deeplab_resnet(num_classes)
#model = pretrained_backbone_tconv_head(device, num_classes)
#print(model)
def model_output_transform(output):
    #print(output["out"].size())
    return output["out"]


# ##############################
# Setup solver
# ##############################

save_every_iters = len(train_loader)

num_epochs = 256

criterion = nn.CrossEntropyLoss()

#lr = 0.007
#lr = 1e-3
#lr = 1e-2
lr = 1e-1
#weight_decay = 5e-4
weight_decay = 1e-1
#optimizer = optim.SGD(
#optimizer = torch.optim.Adam(
optimizer = torch.optim.AdamW(
	[{"params": model.backbone.parameters()}, {"params": model.classifier.parameters()}],
	lr=lr, 
	weight_decay=weight_decay
)

le = len(train_loader)
def lambda_lr_scheduler(iteration, lr0, n, a):
    return lr0 * pow((1.0 - 1.0 * iteration / n), a)

lr_scheduler = lrs.LambdaLR(
    optimizer,
    lr_lambda=[
        partial(lambda_lr_scheduler, lr0=lr, n=num_epochs * le, a=0.9),
        partial(lambda_lr_scheduler, lr0=lr * 1.0, n=num_epochs * le, a=0.9),
        #partial(lambda_lr_scheduler, lr0=lr * 200.0, n=num_epochs * le, a=0.9),
    ],
)




