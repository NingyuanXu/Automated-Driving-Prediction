import pandas as pd
import numpy as np
import os


class dataTransform:

    def __init__(self, features, data_type):
        """[summary]

        Args:
            features (numpy array): list of features eg["x", "y", "present"]
            data_type (str): data type (train, val)
        """
        self.features = features
        self.data_type = data_type
    

    def initialize_model(self, index, X_or_y):
        dataset = np.zeros((10,10,6)) # len(time_step) = 10, num of agents = 10, num of max features = 6   
        file_name = X_or_y + "_" + index + ".csv"
        df = load_dataset(data_type="train", X_or_y="X", filename=file_name)
        X = df.values
        # A0, A1, A2, A3, A4, A5, A6, A7, A8, A9
        agents = np.hsplit(X[:-1, 1:], 10)
        r, c = agents[0].shape
        print(r, c)
        for i in range(len(agents)):
            dataset[:, i, :] = agents[i]
        return dataset

    def get_sample(self, features, time_step, pred_time_step, indices, X_or_y):
        dataset = np.zeros(len(indices))
        for i in range(len(indices)):
            file_name = X_or_y +"_"+ index+ ".csv" # file name "X_0.csv"
            df = self.load_dataset(data_type=self.data_type,
                                   X_or_y=X_or_y, filename=file_name)
            
            

    def load_dataset(self, data_type, X_or_y, filename):
        data_path = data_type
        X_or_y_path = X_or_y
        filename_path = filename
        with open(os.path.join(".", "data", data_path,
             X_or_y_path, filename_path), 'rb') as f:
            return pd.read_csv(f)
