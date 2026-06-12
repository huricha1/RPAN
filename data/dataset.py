from numpy.distutils.conv_template import header
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import pickle
import torch
import numpy as np
import pandas as pd
class Dataset:
    def __init__(self, data, transform=None):

        # Transform
        self.transform = transform

        # load data here
        self.data = data
        self.sampleSize = len(data)
        self.featureSize = data[0].shape[1]

    def return_data(self):
        return self.data


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data[idx]


        if self.transform:
            pass

        return torch.from_numpy(sample)
def get_dataloader(j,window_size=100,batch_size=32):
    # trainData=np.load(f'C4T-RM/SWAT/swat_gap10/swat_gap10_Train.npy',allow_pickle=False)
    trainData_original=pickle.load(
        open(f'E:/IMDiffusion-master/data/Machine/machine-3-{j}_train.pkl', "rb")
    )
    # trainData_original = pickle.load(
    #     open(f'E:/InterFusion-main/data/processed/omi-{j}_train.pkl', "rb")
    # )
    # trainData_original=pd.read_csv(f'E:/JumpStarter-main/dataset/Dataset2/train/service{j}.csv',header=None)
    # trainData_original = pd.read_csv(f'E:/RANSynCoders-main/data/train.csv').drop(columns='timestamp_(min)').values
    train_size = int(0.9 * len(trainData_original))
    trainData=trainData_original[0:train_size]
    validData=trainData_original[train_size:].copy()
    scaler = MinMaxScaler().fit(trainData)
    trainData = scaler.transform(trainData)
    validData = scaler.transform(validData)
    samples_list = list()
    for i in range(0, trainData.shape[0] - window_size + 1):
        samples_list.append(trainData[i:i + window_size])
    # Train data loader
    dataset_train_object = Dataset(data=samples_list, transform=False)
    samples_list = list()
    for i in range(0, validData.shape[0] - window_size + 1,window_size):
        samples_list.append(validData[i:i + window_size])
    # valid data loader
    dataset_valid_object = Dataset(data=samples_list, transform=False)
    samplerRandom = torch.utils.data.sampler.RandomSampler(data_source=dataset_train_object, replacement=True)
    dataloader_train = DataLoader(dataset_train_object, batch_size=batch_size,
                                  shuffle=True, num_workers=0, drop_last=True)
    dataloader_valid = DataLoader(dataset_valid_object, batch_size=batch_size,
                                  shuffle=False, num_workers=0, drop_last=False)
    testData = pickle.load(
        open(f'E:/IMDiffusion-master/data/Machine/machine-3-{j}_test.pkl', "rb")
    )
    # testData = pickle.load(
    #     open(f'E:/InterFusion-main/data/processed/omi-{j}_test.pkl', "rb")
    # )
    # testData = pd.read_csv(f'E:/JumpStarter-main/dataset/Dataset2/test/service{j}.csv', header=None)
    # testData = pd.read_csv(f'E:/RANSynCoders-main/data/test.csv').drop(columns='timestamp_(min)').values
    testData = scaler.transform(testData)
    # print(testData.shape)
    samples_list = list()
    for i in range(0, testData.shape[0]- window_size + 1,window_size):
        samples_list.append(testData[i:i + window_size])

    dataset_test_object = Dataset(data=samples_list, transform=False)

    # samplerRandom = torch.utils.data.sampler.RandomSampler(data_source=dataset_test_object, replacement=True)
    dataloader_test = DataLoader(dataset_test_object, batch_size=batch_size,
                                  shuffle=False, num_workers=0, drop_last=False)
    return dataloader_train, dataloader_valid, dataloader_test
# get_dataloader()