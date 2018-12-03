import os
import time

import torch
import torch.nn.functional as F
from torch import nn, Tensor

class Model(nn.Module):

    def __init__(self):
        super().__init__()
        # TODO: CODE BEGIN
        #raise NotImplementedError
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        # TODO: CODE END

    def forward(self, images: Tensor) -> Tensor:
        # TODO: CODE BEGIN
        #raise NotImplementedError
        images = F.relu(F.max_pool2d(self.conv1(images), 2))
        images = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(images)), 2))
        images = images.view(-1, 320)
        images = F.relu(self.fc1(images))
        images = F.dropout(images, training=self.training)
        images = self.fc2(images)
        return F.log_softmax(images, dim=1)
        # TODO: CODE END

    def loss(self, logits: Tensor, labels: Tensor) -> Tensor:
        # TODO: CODE BEGIN
        #raise NotImplementedError
        loss = F.nll_loss(logits, labels)
        return loss
        # TODO: CODE END

    def save(self, path_to_checkpoints_dir: str, step: int) -> str:
        path_to_checkpoint = os.path.join(path_to_checkpoints_dir,
                                          'model-{:s}-{:d}.pth'.format(time.strftime('%Y%m%d%H%M'), step))
        torch.save(self.state_dict(), path_to_checkpoint)
        return path_to_checkpoint

    def load(self, path_to_checkpoint: str) -> 'Model':
        self.load_state_dict(torch.load(path_to_checkpoint))
        return self
