from magic.factory.optimizer import AdamFactory
from magic.factory.scheduler import LRSchedulerFactory

def check_and_get_model_params(model_class, scheduler_args, experiment_args):
    print('upup')


def check_and_get_scheduler_params(scheduler_class, scheduler_args, experiment_args):
    if isinstance(scheduler_class, LRSchedulerFactory):
        decay = scheduler_args.get('decay', 1)
        gamma = scheduler_args.get('gamma', 0.99)

        experiment_args.decay_period = decay
        experiment_args.gamma = gamma


def check_and_get_optimizer_params(optimizer_class, optimizer_args, experiment_args):
    if isinstance(optimizer_class, AdamFactory):
        b1 = optimizer_args.get('beta1', 0.9)
        b2 = optimizer_args.get('beta2', 0.999)
        weight_decay = optimizer_args.get('weight_decay', 0)       

        experiment_args.beta1 = b1
        experiment_args.beta2 = b2
        experiment_args.weight_decay = weight_decay
