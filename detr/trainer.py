import numpy as np
import torch


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

    def loss(self):
        for number of iterations:
            X, y = next(trainloader)
            y_pred = self.model(X)
            loss = loss_fn(y, y_pred)
            if iter % val_every_iter == 0:
                with torch.no_grad():
                    X_val, y_val = next(valloader)
                    y_vpred = self.model(X_val)
                    loss = loss_fn(y_val, y_vpred)


