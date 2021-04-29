import torch

from datasets import SegmentationDataSet, MoveAxis
from dualPathModel import DualPathModel
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from trainer import Trainer

import os

# root directory, setup by you
root = 'tmp-data-512x512-from-20-pictures'

preprocessor = MoveAxis(True, True)

# input and target files
inputs = list(map(lambda x: os.path.join(os.path.join(root, 'origins'), x), os.listdir(os.path.join(root, 'origins'))))
targets = list(map(lambda x: os.path.join(os.path.join(root, 'classes'), x), os.listdir(os.path.join(root, 'classes'))))

# random seed
random_seed = 42

# split dataset into training set and validation set
train_size = 0.8  # 80:20 split

inputs_train, inputs_valid = train_test_split(
    inputs,
    random_state=random_seed,
    train_size=train_size,
    shuffle=True)

targets_train, targets_valid = train_test_split(
    targets,
    random_state=random_seed,
    train_size=train_size,
    shuffle=True)

# inputs_train, inputs_valid = inputs[:80], inputs[80:]
# targets_train, targets_valid = targets[:80], targets[:80]

# dataset training
dataset_train = SegmentationDataSet(inputs=inputs_train,
                                    targets=targets_train,
                                    transform=preprocessor)

# dataset validation
dataset_valid = SegmentationDataSet(inputs=inputs_valid,
                                    targets=targets_valid,
                                    transform=preprocessor)

# dataloader training
dataloader_training = DataLoader(dataset=dataset_train,
                                 batch_size=2,
                                 shuffle=True)

# dataloader validation
dataloader_validation = DataLoader(dataset=dataset_valid,
                                   batch_size=2,
                                   shuffle=True)

# device
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# model
model = DualPathModel(res_x=512,
                      res_y=512).to(device)

# criterion
criterion = torch.nn.CrossEntropyLoss()
# criterion = DiceLoss()

# optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# trainer
trainer = Trainer(model=model,
                  device=device,
                  criterion=criterion,
                  optimizer=optimizer,
                  training_DataLoader=dataloader_training,
                  validation_DataLoader=dataloader_validation,
                  epochs=10,
                  epoch=0)

# start training
training_losses, validation_losses, lr_rates = trainer.run_trainer()
