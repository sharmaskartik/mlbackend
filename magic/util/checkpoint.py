import os
import magic.util.ensembles as ensembles
import torch
import shutil
import jsonpickle
import copy

def restore_single_checkpoint(experiment, args, logging_handle):

    checkpoint_path = ''

    if args.resume != '':
        checkpoint_path = os.path.join(args.resume, 'checkpoint.pth.tar')

    if os.path.isfile(checkpoint_path):
        logging_handle.info("Loading checkpoint '{}'".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        experiment.model.load_state_dict(checkpoint['state_dict'])
        experiment.optimizer.load_state_dict(checkpoint['optimizer'])
        experiment.scheduler.load_state_dict(checkpoint['scheduler'])
        experiment.begin_epoch = checkpoint['epoch']
        experiment.stats = checkpoint['stats']
        saved_best_models = ensembles.restore_best_models(args.save)
        if saved_best_models:
            experiment.best_models = saved_best_models
        logging_handle.info("Checkpoint loading complete '{}'".format(args.resume))
    else:
        logging_handle.info("No checkpoint found at '{}'".format(args.resume))
        exit()


def save_checkpoint(state, is_best, path):
  filename1 = os.path.join(path, 'checkpoint1.pth.tar')
  filename2 = os.path.join(path, 'checkpoint.pth.tar')
  torch.save(state, filename1)
  shutil.copyfile(filename1, filename2)
  os.remove(filename1)

  if is_best:
    best_filename = os.path.join(path, 'model_best.pth.tar')
    shutil.copyfile(filename2, best_filename)


def restore_multi_checkpoint(experiment, args, logging_handle):

    checkpoint_path = os.path.join(args.resume, 'checkpoint.pth.tar')

    if os.path.isfile(checkpoint_path):
        logging_handle.info("Loading checkpoint '{}'".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)

        experiment.args = jsonpickle.decode(checkpoint['args'])
        experiment.subjects = copy.deepcopy(checkpoint['subjects'])
        experiment.model_factory = checkpoint['model_factory']
        experiment.data_loader_method = checkpoint['data_loader_method']
        experiment.optimizer_factory = checkpoint['optimizer']
        experiment.scheduler_factory = checkpoint['scheduler']
        experiment.criterion = checkpoint['criterion']
        experiment.metric = checkpoint['metric']
        experiment.dataset_location = checkpoint['dataset_location']
        experiment.parent_results_dir = checkpoint['parent_results_dir']

        logging_handle.info("Checkpoint loading complete '{}'".format(args.resume))
    else:
        logging_handle.info("No checkpoint found at '{}'".format(args.resume))
        exit()


def cure_checkpoint(path):
    filename1 = os.path.join(path, 'checkpoint1.pth.tar')
    filename2 = os.path.join(path, 'checkpoint.pth.tar')
    shutil.copyfile(filename1, filename2)
    os.remove(filename1)