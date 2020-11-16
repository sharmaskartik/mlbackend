from __future__ import annotations
import torch
from abc import ABC, abstractmethod

class OptimizerFactory():

    @abstractmethod
    def create_optimizer(self, model, args):
        pass

class AdamFactory(OptimizerFactory):

    def create_optimizer(self,  model, args):
        opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay)
        return opt
