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

    def loss():
        """
        Implement
        """