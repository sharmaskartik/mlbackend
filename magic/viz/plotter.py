import matplotlib.pyplot as plt
import numpy as np
import pickle
from magic.data.stats import SingleSubjectTrainingStats
import os

class PlotStatsForSingleSubject:

    def __init__(self, title, block=False, alpha=0.5, objective_string='Accuracy', save='', show=False, **kwargs):

        super(PlotStatsForSingleSubject, self).__init__()

        self.title = title
        self.block = block
        self.alpha = alpha
        self.objective_string = objective_string
        self.save = save
        self.show = show


        self.plot_info = kwargs.get('fig_and_ax', None)
        if self.plot_info is None:
            self.fig, self.ax = plt.subplots(1, 2, figsize=(10, 10))
        else:
            self.fig, self.ax = self.plot_info[0], self.plot_info[1]

        self.colors = kwargs.get('colors', ['g'] * len(self.list_of_paths_to_stats))
    
        if self.save != '':
            os.makedirs(self.save, exist_ok=True)

    def plot(self, stats):
        train_loss, train_obj, test_loss, test_obj = stats.get_stats()
           

        #check if the results are from multiple runs
        if train_loss.shape[1] > 1:
            self.ax[0].errorbar(np.arange(train_loss.shape[1]), np.mean(train_loss, axis=0),
                                yerr=np.std(train_loss, axis=0), label='Training - Subject %s'%self.title, alpha=self.alpha, color = self.colors[0])

            self.ax[0].errorbar(np.arange(test_loss.shape[1]), np.mean(test_loss, axis=0),
                                yerr=np.std(test_loss, axis=0), label='Testing - Subject %s'%self.title, alpha=self.alpha, color = self.colors[1])

            self.ax[1].errorbar(np.arange(train_obj.shape[1]), np.mean(train_obj, axis=0),
                                yerr=np.std(train_obj, axis=0), label='Training - Subject %s'%self.title, alpha=self.alpha, color = self.colors[0])

            self.ax[1].errorbar(np.arange(test_obj.shape[1]), np.mean(test_obj, axis=0),
                                yerr=np.std(test_obj, axis=0), label='Testing - Subject %s'%self.title, alpha=self.alpha, color = self.colors[1])
        else:

            self.ax[0].plot(train_loss, label = 'Training - Subject %s'%self.title, alpha = self.alpha, color = self.colors[0])
            self.ax[0].plot(test_loss, label = 'Testing - Subject %s'%self.title, alpha = self.alpha, color = self.colors[1])
            self.ax[1].plot(train_obj, label = 'Training - Subject %s'%self.title, alpha = self.alpha, color = self.colors[1])
            self.ax[1].plot(test_obj, label = 'Testing - Subject %s'%self.title, alpha = self.alpha, color = self.colors[0])

        self.ax[0].legend()
        self.ax[1].legend()


        self.ax[0].set_xlabel("epoch")
        self.ax[0].set_ylabel("Loss")
        self.ax[0].set_title('Losses')
        self.ax[1].set_xlabel("epoch")
        self.ax[1].set_ylabel("Objective")
        self.ax[1].set_title(self.objective_string)

        self.fig.suptitle(self.title, fontsize=20)

        if self.show:
            plt.show(block=self.block)
            plt.close()

        if self.save != '':
            image_filename = os.path.join(self.save, str(self.title) + '.png')
            self.fig.savefig(image_filename, dpi=120, bbox_inches='tight')
        
        return self.fig, self.ax



class PlotStatsForMultipleSubjects:
    def __init__(self, path_to_stats, block=False, alpha=0.5, objective_string='Accuracy', show=False, **kwargs):
        super(PlotStatsForMultipleSubjects, self).__init__()
        self.path_to_stats  = path_to_stats
        with open(path_to_stats, 'rb') as f:
            self.stats = pickle.load(f)

        self.block = block
        self.alpha = alpha
        self.objective_string = objective_string
        self.show = show

    def plot(self):
        save_location = os.path.join(os.path.dirname(self.path_to_stats), 'plots')
        os.makedirs(save_location, exist_ok=True)
        for subject, exec_stats_for_subject in self.stats.items():
            stat_obj = SingleSubjectTrainingStats(exec_stats_for_subject)
            plotter = PlotStatsForSingleSubject(subject, self.block, self.alpha, self.objective_string, save=save_location, show=self.show)
            plotter.plot(stat_obj)




class PlotStatsForMultipleExperiments:
    def __init__(self, list_of_paths_to_stats, save, block=False, alpha=0.5, objective_string='BSR', show=False, **kwargs):
        super(PlotStatsForMultipleExperiments, self).__init__()

        self.list_of_paths_to_stats = list_of_paths_to_stats
        self.save = save
        self.block = block
        self.alpha = alpha
        self.objective_string = objective_string
        self.show = show
        self.list_of_pickles = []
        for stat_path in self.list_of_paths_to_stats:
            with open(stat_path, 'rb') as f:
                stats = pickle.load(f)
            self.list_of_pickles.append(stats)
        

        self.colors = kwargs.get('colors', ['b', 'g'])
    
        if self.save != '':
            os.makedirs(self.save, exist_ok=True)

    def plot(self):

        with open(self.list_of_paths_to_stats[0], 'rb') as f:
            stats = pickle.load(f)

        list_of_subjects = [subject for subject in stats.keys()]

        for subject in list_of_subjects:
            self.fig, self.ax = plt.subplots(2, 2, figsize=(10, 10))
            self.title = subject    

            for num_stat, stats in enumerate(self.list_of_pickles):

                exec_stats_for_subject = stats[subject]
                stat_obj = SingleSubjectTrainingStats(exec_stats_for_subject)
                train_loss, train_obj, test_loss, test_obj = stat_obj.get_stats()

                #check if the results are from multiple runs
                if train_loss.shape[1] > 1:
                    self.ax[0][0].errorbar(np.arange(train_loss.shape[1]), np.mean(train_loss, axis=0),
                                        yerr=np.std(train_loss, axis=0), label='Subject %s'%self.title, alpha=self.alpha, color = self.colors[num_stat])

                    self.ax[0][1].errorbar(np.arange(test_loss.shape[1]), np.mean(test_loss, axis=0),
                                        yerr=np.std(test_loss, axis=0), label='Subject %s'%self.title, alpha=self.alpha, color = self.colors[num_stat])

                    self.ax[1][0].errorbar(np.arange(train_obj.shape[1]), np.mean(train_obj, axis=0),
                                        yerr=np.std(train_obj, axis=0), label='Subject %s'%self.title, alpha=self.alpha, color = self.colors[num_stat])

                    self.ax[1][1].errorbar(np.arange(test_obj.shape[1]), np.mean(test_obj, axis=0),
                                        yerr=np.std(test_obj, axis=0), label='Subject %s'%self.title, alpha=self.alpha, color = self.colors[num_stat])
                else:

                    self.ax[0][0].plot(train_loss, label = 'Subject %s'%self.title, alpha = self.alpha, color = self.colors[num_stat])
                    self.ax[0][1].plot(test_loss, label = 'Subject %s'%self.title, alpha = self.alpha, color = self.colors[num_stat])
                    self.ax[1][0].plot(train_obj, label = 'Subject %s'%self.title, alpha = self.alpha, color = self.colors[num_stat])
                    self.ax[1][1].plot(test_obj, label = 'Subject %s'%self.title, alpha = self.alpha, color = self.colors[num_stat])

            self.ax[0][0].legend()
            self.ax[0][1].legend()
            self.ax[1][0].legend()
            self.ax[1][1].legend()


            self.ax[0][0].set_xlabel("epoch")
            self.ax[0][0].set_ylabel("Loss")

            self.ax[0][1].set_xlabel("epoch")
            self.ax[0][1].set_ylabel("Loss")

            self.ax[0][0].set_title('Train Losses')
            self.ax[0][1].set_title('Test Losses')
            
            self.ax[1][0].set_xlabel("epoch")
            self.ax[1][0].set_ylabel("Objective")
            self.ax[1][1].set_xlabel("epoch")
            self.ax[1][1].set_ylabel("Objective")

            self.ax[1][0].set_title('Train %s'%self.objective_string)
            self.ax[1][1].set_title('Test %s'%self.objective_string)
            

            self.fig.suptitle(self.title, fontsize=20)

            if self.show:
                plt.show(block=self.block)
                plt.close()

            if self.save != '':
                image_filename = os.path.join(self.save, str(self.title) + '.png')
                self.fig.savefig(image_filename, dpi=120, bbox_inches='tight')
            
        return self.fig, self.ax