import pandas as pd
import numpy
from os import listdir
from os.path import join

TRAIN_PATH = '../data/car_traj/train/'
VAL_PATH = '../data/car_traj/val/'

DATA_PATH = {'train': TRAIN_PATH,
             'val': VAL_PATH}

# handles getting and transformation of data
class Trajectory:
                        # dataset = train/val
    def __init__(self, dataset = 'train'): 
        path = DATA_PATH[dataset] # gonna throw a genius error
        
        feat_path = path + 'X/'
        targ_path = path + 'y/'

        self.n = len(listdir(feat_path))

        self.features = pd.read_csv(path + 'features.csv')
        self.targets = pd.read_csv(path + 'targets.csv')

    def get_feat(self):
        return self.features

    def get_targets(self):
        return self.targets

    
        
