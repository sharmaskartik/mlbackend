from __future__ import annotations
import torch
from abc import ABC, abstractmethod

class OptimizerFactory():

    @abstractmethod
    def create_optimizer(self, model, args):
        pass

class AdamFactory(OptimizerFactory):

    def create_optimizer(self,  model, args):
        try:
            b1 = args.beta1
        except:
            b1 = 0.9

        try:
            b2 = args.beta2
        except:
            b2 = 0.999
        
        opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(b1, b2), eps=1e-08, weight_decay=args.weight_decay)
        return opt
