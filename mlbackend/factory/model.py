from __future__ import annotations
from mlbackend.model_zoo.schirrmeister import DeepModel
from mlbackend.model_zoo.csu_p300 import p300Model
from mlbackend.model_zoo.eegnet import EEGNet
from abc import ABC, abstractmethod
class ModelFactory():

    @abstractmethod
    def create_model(self):
        pass

class SchirrmeisterDeepModelFactory(ModelFactory):

    def create_model(self, args):
        input_channels = args['input_channels']
        init_channels = args['init_channels']
        num_targets = args['num_targets']
        return DeepModel(input_channels, init_channels, num_targets)


class CSUp300ModelFactory(ModelFactory):

    def create_model(self, args):
        input_channels = args['input_channels']
        init_channels = args['init_channels']
        num_targets = args['num_targets']
        return p300Model(input_channels, init_channels, num_targets)


class EEGNetModelFactory(ModelFactory):

    def create_model(self, args):
        C = args['C']
        F1 = args['F1']
        D = args['D']
        kernel_size = args['kernel_size']
        drop_p = args.get('drop_p', 0.5)

        return EEGNet(C, F1, D, kernel_size, drop_p)


