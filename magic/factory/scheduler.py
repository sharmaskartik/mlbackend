from __future__ import annotations
from abc import ABC, abstractmethod
import torch

class SchedulerFactory():
    @abstractmethod
    def create_scheduler(self, optimizer, args):
        pass

class LRSchedulerFactory(SchedulerFactory):
    def create_scheduler(self, optimizer, args):
        return  torch.optim.lr_scheduler.StepLR(optimizer, args.decay_period, gamma=args.gamma)
