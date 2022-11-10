# %%
import os,sys
from __future__ import print_function, division
import pandas as pd
import numpy as np
import torch 
from torchvision import datasets # load data
from torch.autograd import Variable
import torch.nn.functional as F # implements forward and backward definitions of an autograd operation

# %%
class HIGGS_Dataset(torch.utils.data.Dataset):
    """SUSY pytorch dataset."""

    def __init__(self, dataset_size, root_dir="./Dataset/", train=True, transform=None, high_level_feats=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            train (bool, optional): If set to `True` load training data.
            transform (callable, optional): Optional transform to be applied on a sample.
            high_level_festures (bool, optional): If set to `True`, working with high-level features only. 
                                        If set to `False`, working with low-level features only.
                                        Default is `None`: working with all features
        """

        self.root_dir = root_dir
        
        low_features=["low_level_feature" + str(i) for i in range(10)]

        high_features=["high_level_feature" + str(j) for j in range(15)]

        features = low_features + high_features

        if (dataset_size % 2):
            dataset_size += 1
            print("Need even dataset size. Adjustment was made. New dataset size: ", dataset_size)

        half_size = int(dataset_size/2)
        #Number of datapoints to work with
        df = pd.concat([pd.read_csv(self.root_dir+"htautau.txt", header=None,nrows=half_size,engine='python', sep = '\t', dtype="float32"), pd.read_csv(root_dir+"htautau.txt", header=None,nrows=half_size,engine='python', sep = '\t',dtype="float32")])
        df.columns=features
        labels = [1 for i in range(half_size)] + [0 for i in range(half_size)]
        df["Label"] = labels
        shuffled = df.sample(frac=1, random_state=1).reset_index()
        Y = shuffled['Label']
        X = shuffled[[col for col in df.columns if col!="Label"]]


        # set training, validation and test data size
        train_size=int(.8*dataset_size)
        valid_size=int(.1*dataset_size)
        self.train=train

        if self.train:
            X=X[:train_size]
            Y=Y[:train_size]
            print("Training on {} examples".format(train_size))
        elif self.train == None:
            X=X[train_size:dataset_size-valid_size]
            Y=Y[train_size:dataset_size-valid_size]
            print("Validation on {} examples".format(valid_size))
        else:
            X=X[train_size + valid_size:]
            Y=Y[train_size + valid_size:]
            print("Testing on {} examples".format(valid_size))


        # make datasets using only the 8 low-level features and 10 high-level features
        if high_level_feats is None:
            self.data=(X.values.astype(np.float32), Y.values.astype(int))
            print("Using both high and low level features")
        elif high_level_feats is True:
            self.data=(X[high_features].values.astype(np.float32), Y.values.astype(int))
            print("Using high-level features only.")
        elif high_level_feats is False:
            self.data=(X[low_features].values.astype(np.float32),Y.values.astype(int))
            print("Using low-level features only.")


    # override __len__ and __getitem__ of the Dataset() class

    def __len__(self):
        return len(self.data[1])

    def __getitem__(self, idx):

        sample=(self.data[0][idx,...],self.data[1][idx])

        return sample

# %%
train_loader = torch.utils.data.DataLoader(
        HIGGS_Dataset(10,train=True,high_level_feats=False))

# %%
def load_data(dataset_size, batch_size, high_level_feats):

    train_loader = torch.utils.data.DataLoader(
        HIGGS_Dataset(HIGGS_Dataset(dataset_size,train=True,high_level_feats=high_level_feats),
        batch_size=batch_size, shuffle=True))

    valid_loader = torch.utils.data.DataLoader(
        HIGGS_Dataset(HIGGS_Dataset(dataset_size,train=None,high_level_feats=high_level_feats),
        batch_size=batch_size, shuffle=True))

    test_loader = torch.utils.data.DataLoader(
        HIGGS_Dataset(HIGGS_Dataset(dataset_size,train=False,high_level_feats=high_level_feats),
        batch_size=batch_size, shuffle=True))
    
    return train_loader, test_loader

# %%



