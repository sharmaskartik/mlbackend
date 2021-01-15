import os
from tarfile import RECORDSIZE
from mlbackend.data.stats import SingleSubjectTrainingStats
from mlbackend.util.checkpoint import restore_hyperparameter_checkpoint
import numpy as np
import csv

from numpy.lib.arraysetops import isin, unique

def beautify_hyperparams(subject, hyperparams, metric, returns=['model', 'optimizer', \
                        'scheduler', 'learning_rate', 'epochs', 'time_delay_window_size']):
    result_string = [subject]
    if 'model' in returns:    
        model, model_params = hyperparams['model']
        result_string.extend([model.__class__.__name__, str(model_params)])

    if 'optimizer' in returns:
        optimizer, optimizer_params = hyperparams['optimizer']
        optimizer_params = {k:'%e'%v if isinstance(v,float) else v for k,v in optimizer_params.items()}
        result_string.extend([optimizer.__class__.__name__, str(optimizer_params)])
    
    if 'scheduler' in returns:
        scheduler, scheduler_params = hyperparams['scheduler']
        result_string.extend([scheduler.__class__.__name__, str(scheduler_params)])
    
    if 'learning_rate' in returns:
        result_string.extend(['%e'%hyperparams['learning_rate']])

    if 'epochs' in returns:
        result_string.extend([str(hyperparams['epochs'])])

    if 'time_delay_window_size' in returns:   
        result_string.extend([str(hyperparams['time_delay_window_size'])])

    result_string.extend(['%.3f'%metric])
    return result_string

def get_header(returns):

    result_string = ['subject']
    if 'model' in returns:    
        result_string.extend(['Model', 'Model Params'])

    if 'optimizer' in returns:
        result_string.extend(['Optimizer', 'Optimizer Params'])
    
    if 'scheduler' in returns:
        result_string.extend(['Scheduler', 'Scheduler Params'])
    
    if 'learning_rate' in returns:
        result_string.extend(['Learning Rate'])

    if 'epochs' in returns:
        result_string.extend(['Epochs'])

    if 'time_delay_window_size' in returns:   
        result_string.extend(['Time Delay Window Size'])

    result_string.extend(['Metric'])
    return result_string


def display_hyperparameter_results(path, running_mean_length=10, top_n_params_per_subject=5, sep = '|',\
                         returns=['model', 'optimizer', 'scheduler', 'learning_rate', 'epochs', 'time_delay_window_size']):

    class Dummy: pass
    d = Dummy()
    csv_file = open(os.path.join(path, 'results.csv'), 'w', newline='\n')
    writer = csv.writer(csv_file, delimiter=sep)
    subject_string = 'subject_{}'
    param_string = 'param_%d'
    rep_string = 'rep_%d'

    writer.writerow(get_header(returns))

    restore_hyperparameter_checkpoint(d, path, None)
    for subject in d.subjects:
        subject_dir = subject_string.format(subject)
        results = []
        for param_idx in range(len(d.hyperparams)):
            param_dir =  param_string%param_idx
            for fold in range(d.args.folds):

                rep_dir = rep_string%fold

                stat_dir_path = os.path.join(path, subject_dir, param_dir, rep_dir)
                stat_file = os.path.join(stat_dir_path, 'stat.pickle')
                stat_obj = SingleSubjectTrainingStats(stat_file)
                _, _, _, testing_running_mean = stat_obj.get_running_average(running_mean_length)
                max_mean = np.max(testing_running_mean)
                results.append([param_idx, max_mean])

        
        results  = np.array(results)
        results = results[np.argsort(results[:, 1])[::-1]][:top_n_params_per_subject, :]

        for result in results:
            line = beautify_hyperparams(subject, d.hyperparams[result[0]], result[1], returns=returns)
            writer.writerow(line)
                

def display_hyperparameter_args(path, returns=['model', 'optimizer', 'scheduler', 'learning_rate', 'epochs', 'time_delay_window_size']):
    class Dummy: pass
    d = Dummy()

    restore_hyperparameter_checkpoint(d, path, None)
    unique_args = {}

    for idx, parameters in d.hyperparams.items():
        for param_name, value in parameters.items():
            if param_name not in returns:
                continue
            prev_entry = unique_args.get(param_name, None)
            if prev_entry is None:
                if isinstance(value, list):
                    #list for eg. Model and it's parameters at index 0 and 1 respectively
                    value = value[1]
                    new_value = {}
                    #iterate over all parameters for this hyperparameter and change to list 
                    for k, v in value.items():
                        new_value[k] = [v]
                else:
                    #scalar values like learning rate
                    new_value = [value]

                unique_args[param_name] = new_value
            else:
                if isinstance(value, list):
                    value = value[1]
                    #iterate over all parameters for this hyperparameter
                    # if not present -> add to list 
                    for k, v in value.items():
                        old_list = prev_entry[k]
                        if v not in old_list:
                            old_list.append(v)
                else:
                    if value not in prev_entry:
                        prev_entry.append(value)

    print(unique_args)
    for key, value in unique_args.items():
        print('\n', '-'*70, '\n', key, '\n', '-'*70)
        if isinstance(value, list):
            #parameters like learning rates etc.
            value.sort()
            if key == 'learning_rate':
                value = ['%.3e'%x for x in value]
            print(value)
        else:
            for k, v in value.items():
                print(k)
                v.sort()
                if k == 'weight_decay':
                    v = ['%.3e'%x for x in v]
                print(v)



        

        


if __name__ == '__main__':

    path = '/s/chopin/l/grad/kartikay/code/machine_learning/workspace/bci/bci_experiments/bci_experiments/experiments/csu_p300/logs/train-hyperparameter-search-2020-11-19--09-49-38'
    #display_hyperparameter_results(path, 10, 10,returns=['optimizer', 'learning_rate'])
    display_hyperparameter_args(path, returns=['optimizer', 'learning_rate'])