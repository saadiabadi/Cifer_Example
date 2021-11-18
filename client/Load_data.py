from keras.datasets import cifar10
from keras.utils import to_categorical
import numpy as np
import pickle
import sys
import os
# load dataset

# def partition_dataset(data, nr_of_splits):
#     ind = np.arange(len(data[0]))
#     np.random.shuffle(ind)
#
#     p = np.int32(np.linspace(0, len(ind), nr_of_splits + 1))
#     data_pack = []
#     for d in data:
#         data_pack += [d[ind[p[i]:p[i + 1]]] for i in range(nr_of_splits)]
#     return data_pack


def read_data(path='data'): #, nr_of_splits=1):
    (trainX, trainy), (testX, testy) = cifar10.load_data()
    trainy = to_categorical(trainy)
    testy = to_categorical(testy)

    trainX = trainX.astype('float32')
    testX = testX.astype('float32')
    # normalize to range 0-1
    trainX = trainX / 255.0
    testX = testX / 255.0

    if not os.path.exists(path):
        os.mkdir(path)

    # partition_name = str(nr_of_splits) + "clients"

    # if nr_of_splits > 1:
    #     train_pack = partition_dataset([trainX, trainy], nr_of_splits)
    #     test_pack = partition_dataset([testX, testy], nr_of_splits)
    #
    #     partition_path = os.path.join(path, partition_name)
    #     print("partition_path: ", partition_path)
    #     if not os.path.exists(partition_path):
    #         os.mkdir(partition_path)
    #
    #     for client in range(nr_of_splits):
    #         client_name = "client" + str(client)
    #         client_path = os.path.join(partition_path, client_name)
    #         if not os.path.exists(client_path):
    #             os.mkdir(client_path)
    #
    #         pickle.dump(train_pack[0][client], open(os.path.join(client_path, "trainx.pyp"), "wb"))
    #         pickle.dump(train_pack[1][client], open(os.path.join(client_path, "trainy.pyp"), "wb"))
    #         pickle.dump(test_pack[0][client], open(os.path.join(client_path, "testx.pyp"), "wb"))
    #         pickle.dump(test_pack[1][client], open(os.path.join(client_path, "testy.pyp"), "wb"))
    #
    # else:

    pickle.dump(trainX, open(os.path.join(path, "trainx.pyp"), "wb"))
    pickle.dump(trainy, open(os.path.join(path, "trainy.pyp"), "wb"))
    pickle.dump(testX, open(os.path.join(path, "testx.pyp"), "wb"))
    pickle.dump(testy, open(os.path.join(path, "testy.pyp"), "wb"))

if __name__ == '__main__':

    # nr_of_splits = int(sys.argv[1])
    read_data('data') #, nr_of_splits=nr_of_splits)