
# %%
from datetime import time
import os
import pandas as pd
import numpy as np
from typing import Final


# path to get data csv please do not modify
X_TRAIN: Final = "./data/train/X"
y_TRAIN: Final = "./data/train/y"
X_VAL: Final = "./data/val/X"
y_VAL: Final = "./data/val/y"


# %%
# load dataset
def load_dataset(data_type, X_or_y, filename):
    data_path = data_type
    X_or_y_path = X_or_y
    filename_path = filename
    with open(os.path.join("..", "data",
              data_path, X_or_y_path, filename_path),
         'rb') as f:
        return pd.read_csv(f)


df = load_dataset("train", "X", "X_1.csv")

print(load_dataset("train", "X", "X_1.csv"))

# %%
list = ["1"]
time_step = [-1000, -900, -800]
dataset = np.zeros((len(list), len(time_step), 9, len(features)))
row_filter = np.zeros(len(time_step)).astype(int)
features = ["x", "y", "present"]
width_filter = np.zeros(len(features)).astype(int)   # width filter
feature_filter = {
    "id": 0,
    "role": 1,
    "type": 2,
    "x": 3,
    "y": 4,
    "present": 5
}
for i in range(len(features)):
    width_filter[i] = feature_filter[features[i]]
for i in range(len(time_step)):
    row_filter[i] = int((1000+time_step[i])/100) #(1000-1000)/100 = 0 which is the index of row[-1000, ]
for i in range (len(list)):
    file_name = "X_" + list[i] + ".csv"
    df = load_dataset(data_type="train", X_or_y="X", filename=file_name)
    X = df.values
    columns = df.columns.values
    rows = X[row_filter, :]
    print(rows)
    # dataset[i] = np.zeros(len(time_step))
    # print(dataset)

# %%
list = ["1", "2"]
result_model = np.empty((2, 3, 10, 3),dtype=object)
dataset = np.empty((10, 10, 6), dtype=object)
time_step = [-1000, -900, -800]
row_filter = np.zeros(len(time_step)).astype(int)
for i in range(len(time_step)):
    row_filter[i] = int((1000+time_step[i])/100)
features = ["x", "y", "present"]
width_filter = np.zeros(len(features)).astype(int)   # width filter
feature_filter = {
    "id": 0,
    "role": 1,
    "type": 2,
    "x": 3,
    "y": 4,
    "present": 5
}
for i in range(len(features)):
    width_filter[i] = feature_filter[features[i]]
for i in range(len(list)):
    file_name = "X_" + list[i] + ".csv"
    df = load_dataset(data_type="train", X_or_y="X", filename=file_name)
    X = df.values
    # A0, A1, A2, A3, A4, A5, A6, A7, A8, A9 = split(X[:-1, :1], 10, 6)
    agents = np.hsplit(X[:-1, 1:], 10)

    r,c = agents[0].shape
    print (r,c)
    for j in range (len(agents)):
        dataset[:,j,:] = agents[j]
    rows = dataset[row_filter, :]
    result = rows[:,:, width_filter]
    result_model[i] = result


    


# %%
