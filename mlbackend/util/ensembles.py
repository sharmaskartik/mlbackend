import os
import torch
import heapq

def save_best_models(best_models, save):
    filename = os.path.join(save, 'best_models_for_ensemble.pth.tar')
    torch.save(best_models, filename)

def restore_best_models(save):
    filename = os.path.join(save, 'best_models_for_ensemble.pth.tar')
    if os.path.isfile(filename):
        return torch.load(filename)
    else:
        return None

def update_best_models(best_models, model, objective, n_models_to_save):

    worst_objective = best_models.get('worst', objective)
    model_queue = best_models.get('models', [])

    model_queue, worst_objective = add_to_heap_queue(model.state_dict(), model_queue, worst_objective, objective, n_models_to_save)
    best_models = {'worst': worst_objective, 'models': model_queue}
    return best_models

def add_to_heap_queue(model, model_queue, worst_rank, rank, n_models_to_save):
    if rank >= worst_rank:
        #check if a model with same ranki is already present in the queue
        already_present = False
        for q in model_queue:
            if rank == q[0]:
                already_present = True
                break

        if already_present:
            #don't add, return as it is
            return model_queue, worst_rank

        #remove the model with worst rank if the size gets larger than
        #threshold
        if len(model_queue) >= n_models_to_save:
            #pop the least one
            heapq.heappop(model_queue)

            #update the worst rank
            worst_model_in_queue = heapq.heappop(model_queue)
            worst_rank = worst_model_in_queue[0]
            heapq.heappush(model_queue, worst_model_in_queue)

        #add the current model to the queue
        heapq.heappush(model_queue, (rank, model))

    return model_queue, worst_rank
