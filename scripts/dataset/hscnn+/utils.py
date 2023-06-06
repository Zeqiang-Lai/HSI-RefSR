from __future__ import division

import torch
import torch.nn as nn
from torch.autograd import Variable

import logging
import numpy as np
import os
import hdf5storage

class AverageMeter(object):
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def initialize_logger(file_dir):
    """Print the results in the log file."""
    logger = logging.getLogger()
    fhandler = logging.FileHandler(filename=file_dir, mode='a')
    formatter = logging.Formatter('%(asctime)s - %(message)s',"%Y-%m-%d %H:%M:%S")
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.setLevel(logging.INFO)
    return logger

def save_checkpoint(model_path, epoch, iteration, model, optimizer):
    """Save the checkpoint."""
    state = {
            'epoch': epoch,
            'iter': iteration,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            }
    
    torch.save(state, os.path.join(model_path, 'hscnn_5layer_dim10_%d.pkl' %(epoch)))

def save_matv73(mat_name, var_name, var):
    hdf5storage.savemat(mat_name, {var_name: var}, format='7.3', store_python_metadata=True)






def record_loss(loss_csv,epoch, iteration, epoch_time, lr, train_loss, test_loss):
    """ Record many results."""
    loss_csv.write('{},{},{},{},{},{}\n'.format(epoch, iteration, epoch_time, lr, train_loss, test_loss))
    loss_csv.flush()    
    loss_csv.close

def get_reconstruction(input, num_split, dimension, model):
    """As the limited GPU memory split the input."""
    input_split = torch.split(input,  int(input.shape[3]/num_split), dim=dimension)
    output_split = []
    for i in range(num_split):
        var_input = Variable(input_split[i].cuda(),volatile=True)
        var_output = model(var_input)
        output_split.append(var_output.data)
        if i == 0:
            output = output_split[i]
        else:
            output = torch.cat((output, output_split[i]), dim=dimension)
    
    return output

def reconstruction(rgb,model):
    """Output the final reconstructed hyperspectral images."""
    img_res = get_reconstruction(torch.from_numpy(rgb).float(),1, 3, model)
    img_res = img_res.cpu().numpy()*4095
    img_res = np.transpose(np.squeeze(img_res))
    img_res_limits = np.minimum(img_res,4095)
    img_res_limits = np.maximum(img_res_limits,0)
    return img_res_limits

def rrmse(img_res,img_gt):
    """Calculate the relative RMSE"""
    error= img_res- img_gt
    error_relative = error/img_gt
    rrmse = np.mean((np.sqrt(np.power(error_relative, 2))))
    return rrmse
