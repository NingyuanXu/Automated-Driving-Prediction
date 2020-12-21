import numpy as np
import pandas as pd
from mlp import NeuralNet
from sklearn.neural_network import MLPRegressor


def L2_norm(x1, x2, y1, y2):
    return ((x1 - y2) ** 2 + (y1 - y2) ** 2) ** 0.5


def find_agent(x1):
    n = 5
    k = 3 + 5 * 4
    matrix = np.zeros((11, k))
    for i in range(61):
        if x1[:, i][1] == ' agent':
            matrix[:, 0] = x1[:, i + 2]
            matrix[:, 1] = x1[:, i + 3]
            matrix[:, 2] = x1[:, i + 4]
    return matrix


def find_dist(x1, matrix):
    dist = np.zeros((11, 10))

    for i in range(9):
        k = i * 6 + 4
        dist[:, i] = L2_norm(x1[:, k], x1[:, k + 1], matrix[:, 0], matrix[:, 1])
    #     print("k",dist.shape)
    return dist


def get_small(dist):
    small_index = np.zeros((11, 5))
    #     print("dist",dist)
    for i in range(11):
        cur_dist = dist[i]
        k = 6
        result = np.argsort(cur_dist)
        #         print("small_index",result[1:k])
        small_index[i] = result[1:k]

    return small_index


def complete_matrix(x1):
    matrix = find_agent(x1)
    dist = find_dist(x1, matrix)
    small_index = get_small(dist)
    N, D = small_index.shape
    #     print(len(small_index))
    for j in range(N):
        for i in range(D):

            raw = small_index[j][i]

            true = int(raw * 6 + 4)

            matrix[j][3 + i * 4] = x1[j][true]
            #             print(i,matrix[j,3+i*2])
            matrix[j][4 + i * 4] = x1[j][true + 1]
            matrix[j][5 + i * 4] = x1[j][true + 2]
            #         print("x1",x1[:,true-1])
            if x1[0][true - 1] == ' car':
                matrix[j][6 + i * 4] = 1
            else:
                matrix[j][6 + i * 4] = 0

    return matrix


def compute_matrix(newdata, n):
    result = []
    for i in range(n):
        x = newdata[i]
        temp = complete_matrix(x)
        temp = temp.flatten()
        result.append(temp)
    # print(result)

    result = np.asarray(result)
    return result


def compute_error(matrix1, matrix2):
    size = matrix1.shape[0] * matrix1.shape[1]
    diff = np.sum(np.square(matrix1 - matrix2)) / size
    return np.sqrt(diff)


if __name__ == '__main__':

    data = pd.read_csv('Xtrain_combined.csv', low_memory=False)
    data.head()
    n=np.shape(data)[1]
    newdata=data.values
    N,D = newdata.shape
    n = int(N/11)
    newdata = np.array_split(newdata, n)
    Xtrain =compute_matrix(newdata,n)

    data = pd.read_csv('Xvalidate_combined.csv', low_memory=False)
    data.head()
    n=np.shape(data)[1]
    newdata=data.values
    N,D = newdata.shape
    n = int(N/11)
    newdata = np.array_split(newdata, n)
    Xval =compute_matrix(newdata,n)
    # print(Xval, Xval.shape)

    data = pd.read_csv('Xtest_combined.csv', low_memory=False)
    data.head()
    n=np.shape(data)[1]
    newdata=data.values
    N,D = newdata.shape
    n = int(N/11)
    newdata = np.array_split(newdata, n)
    Xtest =compute_matrix(newdata,n)

    y = pd.read_csv('ytrain_combined.csv')
    y = y.values
    index_list = np.where(y[:,0] == 3000)[0]
    N,D = y.shape
    y_train = np.zeros((2297,60))
    numitr = 2308
    out_index = 0
    for i in index_list:
        matrix = y[i-29:i+1,1:]
        matrix = matrix.flatten()
        y_train[out_index] = matrix
        out_index = out_index+1

    x_out = np.zeros((2297,99))
    place_dist = []
    ptr1 = 0
    ptr2 = 0
    n,d = x_out.shape
    org_list = np.where(y[:,0] == 100)[0]
    while ptr1 < 2308:
        if y[org_list[ptr1]][1] == y_train[ptr2][0]:
            ptr1 = ptr1+1
            ptr2 = ptr2+1
        else:
            place_dist.append(ptr1)
            ptr1 = ptr1+1
    place_dist = np.array(place_dist)
    x_out = np.delete(Xtrain, place_dist, 0)
    Xtrain = x_out

    y = pd.read_csv('yvalidate_combined.csv')
    y = y.values
    index_list = np.where(y[:,0] == 3000)[0]
    l = len(index_list)
    N,D = y.shape
    y_validate = np.zeros((l, 60))
    numitr = Xval.shape[0]
    out_index = 0
    for i in index_list:
        matrix = y[i-29:i+1,1:]
        matrix = matrix.flatten()
        y_validate[out_index] = matrix
        out_index = out_index+1

    x_out2 = np.zeros((l,99))
    place_dist = []
    ptr1 = 0
    ptr2 = 0
    n,d = x_out2.shape
    org_list = np.where(y[:,0] == 100)[0]
    while ptr1 < numitr:
        if y[org_list[ptr1]][1] == y_validate[ptr2][0]:
            ptr1 = ptr1+1
            ptr2 = ptr2+1
        else:
            place_dist.append(ptr1)
            ptr1 = ptr1+1
    place_dist = np.array(place_dist)
    x_out2 = np.delete(Xval, place_dist, 0)
    Xval = x_out2

    print(Xtrain.shape, y_train.shape, Xval.shape, y_validate.shape, Xtest.shape)

    # hidden_layer_choice = [50, 50]
    # iteration_choice = [100, 200, 300, 400, 500]
    # lammy_choice = [1,0.1,0.01,0.001,0.0001]
    #
    # lowest_err = np.inf
    # best_param = []
    # for i in iteration_choice:
    #     for l in lammy_choice:
    #         model = NeuralNet(hidden_layer_sizes=hidden_layer_choice, max_iter=i, lammy=l)
    #         model.fit(Xtrain, y_train)
    #         y_pred_val = model.predict(Xval)
    #         val_err = compute_error(y_validate, y_pred_val)
    #         if val_err < lowest_err:
    #             lowest_err = val_err
    #             best_param = [i, l]
    # print(lowest_err, best_param)

    model = NeuralNet(hidden_layer_sizes=[5], max_iter=100, lammy=0.1)
    model.fit(Xtrain, y_train)
    y_pred_val = model.predict(Xval)
    val_err = compute_error(y_pred_val, y_validate)
    print(val_err)
    y_pred = model.predict(Xtest)
    y_final = y_pred.flatten().T
    pd.DataFrame(y_final).to_csv("./result.csv", header=None, index=None)