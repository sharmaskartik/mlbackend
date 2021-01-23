import pickle
import os
import numpy as np

class SingleSubjectTrainingStats():
    def __init__(self, path=None):
        super(SingleSubjectTrainingStats, self).__init__()
        if path is None:

            self.training_losses = np.array([]).reshape(-1)
            self.training_objectives = np.array([]).reshape(-1)
            self.validation_losses = np.array([]).reshape(-1)
            self.validation_objectives = np.array([]).reshape(-1)

            self.testing_losses = np.array([]).reshape(-1)
            self.testing_objectives = np.array([]).reshape(-1)

        else:
            #read stat object
            with open(path, 'rb') as f:
                obj = pickle.load(f)
            self.training_losses = obj['training_losses']
            self.training_objectives = obj['training_objectives']
            self.testing_losses = obj['testing_losses']
            self.testing_objectives = obj['testing_objectives']
            self.validation_losses = obj.get('validation_losses', None)
            self.validation_objectives = obj.get('validation_objectives', None)

    def update_stats(self, training_loss, training_objective, testing_loss=None, testing_objective=None, valid_loss=None, valid_objective=None):

        self.training_losses = np.append(self.training_losses, training_loss)
        self.training_objectives = np.append(self.training_objectives, training_objective)
        self.testing_losses = np.append(self.testing_losses, testing_loss)
        self.testing_objectives = np.append(self.testing_objectives, testing_objective)

        if valid_loss is not None:
            self.validation_losses = np.append(self.validation_losses, valid_loss)
            self.validation_objectives = np.append(self.validation_objectives, valid_objective)


    def append_stats_from_another_run(self, stat2):
        self.training_losses = np.vstack((self.training_losses, stat2.training_losses.reshape(-1)))
        self.training_objectives = np.vstack((self.training_objectives, stat2.training_objectives.reshape(-1)))

        self.testing_losses = np.vstack((self.testing_losses, stat2.testing_losses.reshape(-1)))
        self.testing_objectives = np.vstack((self.testing_objectives, stat2.testing_objectives.reshape(-1)))

        self.validation_losses = np.vstack((self.validation_losses, stat2.validation_losses.reshape(-1)))
        self.validation_objectives = np.vstack((self.validation_objectives, stat2.validation_objectives.reshape(-1)))

    def get_running_average(self, n):
        training_loss = np.convolve(self.training_losses, np.ones(n)/n, mode='valid')
        training_objectives = np.convolve(self.training_objectives, np.ones(n)/n, mode='valid')
        testing_losses = np.convolve(self.testing_losses, np.ones(n)/n, mode='valid')
        testing_objectives = np.convolve(self.testing_objectives, np.ones(n)/n, mode='valid')
        if self.validation_losses.shape[0] == 0:
            return training_loss, training_objectives, testing_losses, testing_objectives
        else:
            validation_losses = np.convolve(self.validation_losses, np.ones(n)/n, mode='valid')
            validation_objectives = np.convolve(self.validation_objectives, np.ones(n)/n, mode='valid')
            return training_loss, training_objectives, validation_losses, validation_objectives, testing_losses, testing_objectives

    def get_stats(self):
        return self.training_losses, self.training_objectives, \
            self.testing_losses, self.testing_objectives

    def save(self, directory):
        with open(os.path.join(directory, 'stat.pickle'), 'wb') as f:
            pickle.dump(self.__dict__, f)

class MultiTrainingStats():
    def __init__(self, obj=None):
        super(MultiTrainingStats, self).__init__()
        self.subjects = {}

    def add_stat_for_subject(self, subject, stat_file_path):
        #read stat object
        with open(stat_file_path, 'rb') as f:
            stat_new = pickle.load(f)

        stat_existing = self.subjects.get(subject, None)

        if stat_existing is None:
            self.subjects[subject] = SingleSubjectTrainingStats(obj=stat_new)
        else:
            #self.subjects[subject] = SingleSubjectTrainingStats(obj=stat_new)
            stat_existing.append_stats_from_another_run(SingleSubjectTrainingStats(obj=stat_new))

    def save(self, directory):
        save_dict = {}
        for k, v in self.subjects.items():
            save_dict[k] = v.__dict__
        with open(os.path.join(directory, 'stat.pickle'), 'wb') as f:
            pickle.dump(save_dict, f)