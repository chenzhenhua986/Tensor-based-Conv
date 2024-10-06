import torch
from torch import optim, nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision.utils import make_grid
from torchvision import transforms as T
from torchvision import models, datasets
from torchsummary import summary
import numpy as np
#import albumentations as A

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, Precision, Recall, mIoU, IoU, DiceCoefficient, confusion_matrix
from ignite.handlers import LRScheduler, ModelCheckpoint, global_step_from_engine
from ignite.contrib.handlers import ProgressBar, TensorboardLogger
import ignite.contrib.engines.common as common

from PIL import Image
import cv2
import os
from random import randint
from net import tconv1

class Net(nn.Module):
    def __init__(self, device):
        super(Net, self).__init__()     
        self.model = tconv1(device, 23) 

    def forward(self, x):
        x=self.model(x)
        return x


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:2" if use_cuda else "cpu")

t_train_val = T.Compose([
                T.Resize((64, 64)), # Resize images to 256 x 256
                T.ToTensor(),  # Converting cropped images to tensors
                T.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
])

t_target = T.Compose([
                T.ToTensor(),  # Converting cropped images to tensors
                T.Resize((64, 64)), # Resize images to 256 x 256
])

batch_size = 16

trainset = datasets.Cityscapes(root='./data/cityscapes', split='train', mode='fine', target_type='semantic', transform=t_train_val, target_transform=t_target)
#trainset = datasets.Cityscapes(root='./data/cityscapes', split='train', mode='fine', target_type='semantic', transforms=t_train_val)
train_loader_pretrain = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
testset = datasets.Cityscapes(root='./data/cityscapes', split='val', mode='fine', target_type='semantic', transform=t_train_val, target_transform=t_target)
#testset = datasets.Cityscapes(root='./data/cityscapes', split='val', mode='fine', target_type='semantic', transforms=t_train_val)
val_loader_pretrain = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

#img, smnt = dataset[0]

model = Net(device)
model = model.to(device)
#summary(model, (3, 32, 32), batch_size=16)

# Define hyperparameters and settings
#lr = 0.001  # Learning rate
lr = 1e-3  # Learning rate
num_epochs = 8  # Number of epochs
log_interval = 300  # Number of iterations before logging

# Set loss function (categorical Cross Entropy Loss)
loss_func = nn.CrossEntropyLoss()

# Set optimizer (using Adam as default)
optimizer = optim.Adam(model.parameters(), lr=lr)


# Setup pytorch-ignite trainer engine
trainer = create_supervised_trainer(model, optimizer, loss_func, device=device)

# Add progress bar to monitor model training
ProgressBar(persist=True).attach(trainer, output_transform=lambda x: {"Batch Loss": x})

# Define evaluation metrics
cm = confusion_matrix.ConfusionMatrix(num_classes=3)
metrics = {
    "IoU": IoU(cm), 
    "DiceCoefficient": DiceCoefficient(cm), 
    "mIoU": mIoU(cm), 
    "accuracy": Accuracy(), 
    "loss": Loss(loss_func),
}

# Evaluator for training data
train_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)

# Evaluator for validation data
evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)

# Display message to indicate start of training
@trainer.on(Events.STARTED)
def start_message():
    print("Begin training")

# Log results from every batch
@trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
def log_batch(trainer):
    batch = (trainer.state.iteration - 1) % trainer.state.epoch_length + 1
    print(f"Epoch {trainer.state.epoch} / {num_epochs}, "
          f"Batch {batch} / {trainer.state.epoch_length}: "
          f"Loss: {trainer.state.output:.3f}")

# Evaluate and print training set metrics
@trainer.on(Events.EPOCH_COMPLETED)
def log_training_loss(trainer):
    print(f"Epoch [{trainer.state.epoch}] - Loss: {trainer.state.output:.2f}")
    train_evaluator.run(train_loader_pretrain)
    epoch = trainer.state.epoch
    metrics = train_evaluator.state.metrics
    print(f"Train - Loss: {metrics['loss']:.3f}, "
          f"Accuracy: {metrics['accuracy']:.3f} ")

# Evaluate and print validation set metrics
@trainer.on(Events.EPOCH_COMPLETED)
def log_validation_loss(trainer):
    evaluator.run(val_loader_pretrain)
    epoch = trainer.state.epoch
    metrics = evaluator.state.metrics
    print(f"Validation - Loss: {metrics['loss']:.3f}, "
          f"Accuracy: {metrics['accuracy']:.3f}")

# Sets up checkpoint handler to save best n model(s) based on validation accuracy metric
common.save_best_model_by_val_score(
          output_path="best_models",
          evaluator=evaluator, model=model,
          metric_name="accuracy", n_saved=1,
          trainer=trainer, tag="seg")

trainer.run(train_loader_pretrain, max_epochs=num_epochs)
print(evaluator.state.metrics)

