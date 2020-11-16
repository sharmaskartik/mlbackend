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
  filename = os.path.join(path, 'checkpoint.pth.tar')
  torch.save(state, filename)
  if is_best:
    best_filename = os.path.join(path, 'model_best.pth.tar')
    shutil.copyfile(filename, best_filename)


def restore_multi_checkpoint(experiment, args, logging_handle):

    checkpoint_path = os.path.join(args.resume, 'checkpoint.pth.tar')

    if os.path.isfile(checkpoint_path):
        logging_handle.info("Loading checkpoint '{}'".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)

        experiment.args = jsonpickle.decode(checkpoint['args'])
        experiment.dataset_location = checkpoint['dataset_location']
        experiment.data_loader_method = checkpoint['data_loader_method']

        experiment.subjects = copy.deepcopy(checkpoint['all_subjects'])
        experiment.start_iter = checkpoint['last_rep']
        experiment.model_factory = checkpoint['model']
        experiment.optimizer_factory = checkpoint['optimizer']
        experiment.scheduler_factory = checkpoint['scheduler']
        experiment.criterion = checkpoint['criterion']
        experiment.parent_results_dir = checkpoint['parent_results_dir']
        experiment.remaining_subjects = checkpoint['remaining_subjects']
        last_subject = checkpoint['last_subject']

        #if the last epoch was the final epoch, all iterations for the last subject are done
        #remove the subject
        if experiment.start_iter == experiment.args.reps_per_subject:
            experiment.remaining_subjects.pop(last_subject)
            experiment.start_iter = 0

        saved_best_models = ensembles.restore_best_models(args.save)
        if saved_best_models:
            experiment.best_models = saved_best_models
        logging_handle.info("Checkpoint loading complete '{}'".format(args.resume))
    else:
        logging_handle.info("No checkpoint found at '{}'".format(args.resume))
        exit()