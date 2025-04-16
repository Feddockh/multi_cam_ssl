import numpy as np
import torch
import HungarianMatcher

class Trainer(object):
    def __init__(self, model, train_dataloader, val_dataloader, learning_rate = 0.001, 
                 num_epochs = 10, print_every = 10, verbose = True, device = 'cuda'):
      
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.print_every = print_every
        self.verbose = verbose 
        self.loss_history = []
        self.val_loss_history = []
        self.device = device
        self.optim = torch.optim.Adam(self.model.parameters(), self.learning_rate)
        self.matcher = HungarianMatcher()


    def loss(self, outputs, target):
        """
        outputs: This is a dict that contains at least these entries:
     "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
     "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
     "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
               objects in the target) containing the class labels
     "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
        """
        indices = self.matcher(outputs, target)
        l1_loss = nn.l1loss(outputs['bbox'][indices], target['bbox'][indices])
        pred_loss = nn.celoss(outputs['label'][indices], target['label'][indices])
        return l1_loss + pred_loss

    def train(self):
        for number of iterations:
            X, y = next(trainloader)
            y_pred = self.model(X)
            loss = self.loss(y, y_pred)

            if iter % val_every_iter == 0:
                X_val, y_val = next(valloader)
                y_vpred = self.model.predict(X_val)
                loss = self.loss(y_val, y_vpred)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


