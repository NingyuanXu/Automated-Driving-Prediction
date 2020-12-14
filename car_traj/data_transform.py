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
        self.feature_filter = {
            "id": 0,
            "role": 1,
            "type": 2,
            "x": 3,
            "y": 4,
            "present": 5
        }

    def initialize_model(self, index, X_or_y):
        # height: len(time_step) = 10,
        # length: num of agents = 10,
        # width: num of max features = 6
        dataset = np.empty((10, 10, 6), dtype=object)
        file_name = X_or_y + "_" + index + ".csv"
        df = self.load_dataset(data_type="train", X_or_y="X", filename=file_name)
        X = df.values
        # A0, A1, A2, A3, A4, A5, A6, A7, A8, A9
        agents = np.hsplit(X[:-1, 1:], 10)
        r, c = agents[0].shape
        print(r, c)
        for i in range(len(agents)):
            dataset[:, i, :] = agents[i]
        return dataset

    def get_sample(self, features, time_step, pred_time_step, indices, X_or_y):
        # initialize sample dataset
        dataset = np.empty((len(indices), len(time_step),
                            10, len(features)), dtype=object)
        pred_dataset = np.empty((len(indices), len(time_step),
                                 10, len(features)), dtype=object)

        # get the filter based on features, time_step
        height_filter = np.zeros(len(time_step)).astype(int)  # height filter
        for i in range(len(time_step)):
            height_filter[i] = int((1000+time_step[i])/100)

        width_filter = np.zeros(len(features)).astype(int)   # width filter
        for i in range(len(features)):
            width_filter[i] = self.feature_filter[features[i]]

        pred_height_filter = np.zeros(len(time_step)).astype(int)  # height filter
        for i in range(len(time_step)):
            pred_height_filter[i] = int((1000+time_step[i])/100)
        for i in range(len(indices)):
            csv_model = self.initialize_model(index=indices[i], X_or_y=X_or_y)
            # Get sample model
            buffer_model = csv_model[height_filter, :]
            result_model = buffer_model[:,:, width_filter]
            dataset[i] = result_model
            # Get target pred sample model
            pred_buffer_model = csv_model[pred_height_filter, :]
            pred_result_model = pred_buffer_model[:, :, width_filter]
            pred_dataset[i] = pred_result_model
        return dataset, pred_dataset

    def load_dataset(self, data_type, X_or_y, filename):
        data_path = data_type
        X_or_y_path = X_or_y
        filename_path = filename
        with open(os.path.join(".", "data", data_path, X_or_y_path, filename_path), 'rb') as f:
            return pd.read_csv(f)
