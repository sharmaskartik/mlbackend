import time
import os
import numpy as np
import sys
import jsonpickle
import copy
import ray
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from magic.util.checkpoint import  restore_single_checkpoint, restore_multi_checkpoint, save_checkpoint, cure_checkpoint
from magic.util.logger import get_logging_handle
from magic.data.stats import SingleSubjectTrainingStats, MultiTrainingStats
import magic.util.ensembles as ensembles
import magic.util.util as util
from magic.util.io_helper import create_exp_dir


ray.init()

@ray.remote(num_gpus=0.05)
class SingleSubjectExperiment(object):
    def __init__(self, model_factory, data_loader_method, optimizer_factory, scheduler_factory, criterion, metric, args):

        self.model_factory = model_factory
        self.optimizer_factory = optimizer_factory
        self.scheduler_factory = scheduler_factory
        self.data_loader_method = data_loader_method

        self.args = args
        self.begin_epoch = 0
        self.stats = SingleSubjectTrainingStats()
        self.metric = metric
        self.criterion = criterion

        train_queue, valid_queue = self.data_loader_method(self.args.dataset_location, self.args.data_filename, self.args)
        self.train_queue = train_queue
        self.valid_queue = valid_queue
        x, l = next(iter(train_queue))
        model_args = self.args.model_args
        model_args['input_channels'] = x.shape[2]
        self.model = self.model_factory.create_model(model_args) 
        self.optimizer = self.optimizer_factory.create_optimizer(self.model, self.args)
        self.scheduler = self.scheduler_factory.create_scheduler(self.optimizer, self.args)
        self.setup()

        if self.args.resume != '':
            restore_single_checkpoint(self, self.args, self.logging_handle)


    def setup(self):

        if self.args.resume != '':
            self.args.save = self.args.resume
        else:
            if not self.args.multi_subject_experiment:
                self.args.save = '{}-{}'.format(self.args.save, time.strftime("%Y-%m-%d--%H-%M-%S"))
                
            self.args.save = os.path.join(self.args.results_dir, self.args.save)


        create_exp_dir(self.args.save)
        self.logging_handle = get_logging_handle(self.args.save)

        if not torch.cuda.is_available():
            self.logging_handle.info('no gpu device available')
            sys.exit(1)

        self.model.cuda()
        np.random.seed(self.args.seed)
        torch.cuda.set_device(self.args.gpu)
        #cudnn.benchmark = True
        torch.manual_seed(self.args.seed)
        #cudnn.enabled=True
        torch.cuda.manual_seed(self.args.seed)

        self.best_models = {}

    def run(self):
        best_loss = 0

        for epoch in range(self.begin_epoch, self.args.epochs):

            self.logging_handle.info('epoch %d lr %e', epoch, self.scheduler.get_lr()[0])

            train_objective, train_loss = self._train()
            self.logging_handle.info('train_acc \t %e \t ' + '\t %f \t' , train_loss, train_objective)

            valid_objective, valid_loss = self._infer()
            self.logging_handle.info('valid_acc \t %e \t ' + '\t %f \t' , valid_loss, valid_objective)
            self.scheduler.step()

            self.stats.update_stats(train_loss, train_objective, valid_loss, valid_objective)

            is_best = False
            best_validation_obj = -1
            if valid_objective > best_validation_obj:
                best_validation_obj = valid_objective
                is_best = True

            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'best_valid_obj': best_validation_obj,
                'optimizer' : self.optimizer.state_dict(),
                'scheduler' : self.scheduler.state_dict(),
                'stats' : self.stats
                }, is_best, self.args.save)

            ensembles.save_best_models(self.best_models, self.args.save)

            self.stats.save(self.args.save)

        return self.stats

    def _train(self):

        losses = util.AvgrageMeter()
        objective = util.AvgrageMeter()

        self.model.train()

        for step, (input, targets) in enumerate(self.train_queue):

            targets = targets.cuda(non_blocking=True)
            input = input.cuda()
            input = Variable(input.squeeze(1))
            targets = Variable(targets).view(-1)

            self.optimizer.zero_grad()
            logits = self.model(input)

            loss = self.criterion(logits, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
            self.optimizer.step()

            obj = self.metric(logits, targets)

            n = input.size(0)
            losses.update(loss.item(), n)
            objective.update(obj[0].item(), n)

            if step % self.args.report_freq == 0:
                self.logging_handle.info('train %04d \t %e \t' +'\t %f \t' , step, losses.avg, objective.avg )


        return objective.avg, losses.avg

    def _infer(self):

        self.model.eval()

        losses = util.AvgrageMeter()
        objective = util.AvgrageMeter()

        for step, (input, targets) in enumerate(self.valid_queue):

            with torch.no_grad():
                input = Variable(input.squeeze(1)).cuda()
                targets = Variable(targets).cuda(non_blocking=True).view(-1)
                logits = self.model(input)

                n = input.size(0)
                loss = self.criterion(logits, targets)
                obj = self.metric(logits, targets)

                losses.update(loss.item(), n)
                objective.update(obj[0].item(), n)
                if step % self.args.report_freq == 0:
                    self.logging_handle.info('valid  %04d \t %e \t ' +  '\t %f \t' , step, losses.avg,  objective.avg)


        self.best_models = ensembles.update_best_models(self.best_models, self.model, objective.avg, self.args.n_models_to_save)
        return  objective.avg, losses.avg


class Experiment():
    def __init__(self, dataset_location, subjects, data_loader_method,  model_factory, optimizer_factory, scheduler_factory, criterion, metric, args):
        super(Experiment, self).__init__()
        self.args = args
        self.setup()
        if args.resume != '':
            restore_multi_checkpoint(self, self.args, self.logging_handle)

        else:
            self.subjects = subjects
            self.model_factory = model_factory
            self.data_loader_method = data_loader_method
            self.optimizer_factory = optimizer_factory
            self.scheduler_factory = scheduler_factory
            self.criterion = criterion
            self.metric = metric
            self.dataset_location = dataset_location


            #save checkpoint
            save_checkpoint({
                'args': jsonpickle.encode(self.args),
                'subjects': self.subjects,
                'model_factory': self.model_factory,
                'data_loader_method':self.data_loader_method,
                'optimizer': self.optimizer_factory,
                'scheduler': self.scheduler_factory,
                'criterion': self.criterion,
                'metric' : metric,
                'dataset_location':self.dataset_location,
                'parent_results_dir':self.parent_results_dir,
            }, False, self.parent_results_dir)

        self.experiments = {}
        for subject in self.subjects:
            details_for_subject = {}
            subject_dir = 'subject_{}'.format(subject)
            for rep in range(self.args.reps_per_subject):

                rep_dir = 'rep_%d' % rep
                checkpoint_dir_path = os.path.join(self.parent_results_dir, subject_dir, rep_dir)
                path_to_checkpoint = os.path.join(checkpoint_dir_path, 'checkpoint.pth.tar')
                #load checkpoint to check status
                if os.path.exists(path_to_checkpoint):
                    try:
                        checkpoint = torch.load(path_to_checkpoint)
                    except:
                        #check if checkpoint1.pth.tar exists -> if it does, it means that there was a crash when writing the file
                        #restore the correct file
                        if os.path.exists(os.path.join(checkpoint_dir_path, 'checkpoint1.pth.tar')):
                            cure_checkpoint(checkpoint_dir_path)
                            try:
                                checkpoint = torch.load(path_to_checkpoint)
                            except:
                                #checkpoint file is corrupt
                                details_for_subject[rep_dir] = ''
                                continue
                        else:
                            #checkpoint file is corrupt
                            details_for_subject[rep_dir] = ''
                            continue

                    #check if the training had finished
                    if checkpoint['epoch'] < self.args.epochs:
                        details_for_subject[rep_dir] = checkpoint_dir_path  
                else:
                    details_for_subject[rep_dir] = ''
                
            if len(details_for_subject) > 0:
                self.experiments[subject] = details_for_subject


        self.stats = MultiTrainingStats()

    def setup(self):

        if self.args.resume != '':
            self.args.results_dir = self.args.resume
        else:
            self.args.save = '{}-{}'.format('train-multi-subject', time.strftime("%Y-%m-%d--%H-%M-%S"))
            self.args.results_dir = os.path.join(self.args.results_dir, self.args.save)
        self.parent_results_dir = self.args.results_dir

        create_exp_dir(self.args.results_dir)
        self.logging_handle = get_logging_handle(self.args.results_dir)

    def run(self):
        actors = []
        self.logging_handle.info('Beginning the run method')
        for subject, experiments in self.experiments.items():

            data_filename = self.subjects[subject]
            self.args.results_dir = os.path.join(self.parent_results_dir, 'subject_{}'.format(subject))

            self.args.dataset_location = self.dataset_location
            self.args.data_filename = data_filename
            for key, resume in experiments.items():
                self.args.resume = resume
                self.args.save = key
                exp = SingleSubjectExperiment.remote(self.model_factory, self.data_loader_method, self.optimizer_factory, self.scheduler_factory,  self.criterion, self.metric, copy.copy(self.args))
                actors.append(exp)
        
        n_actors = len(actors)
        n_finished = 0
        n_actors_to_run = 20
        futures = []

        def run_actors(actors, futures, n_actors_to_run):
            for i, actor in enumerate(actors):
                futures.append(actor.run.remote())
                actors.remove(actor)
                if i == n_actors_to_run - 1:
                    break

        run_actors(actors, futures, n_actors_to_run)

        while n_finished < n_actors:
            finished, running = ray.wait(futures)#, num_returns = len(futures))
            futures = running
            n_finished += len(finished)
            run_actors(actors, futures, len(finished))


        self.logging_handle.info('All models have finished training.')
        #re run loops to collect all the data together
        for subject, file_name in self.subjects.items():
            self.args.results_dir = os.path.join(self.parent_results_dir, 'subject_{}'.format(subject))
            for rep in range(self.args.reps_per_subject):
                self.args.save = 'rep_%d' % rep
                stat_file_path = os.path.join(self.args.results_dir, self.args.save, 'stat.pickle')
                self.stats.add_stat_for_subject(subject, stat_file_path)

        self.stats.save(self.parent_results_dir)
        self.args.resume = self.parent_results_dir
        return self.stats

