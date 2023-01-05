import torch
import pandas as pd
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import math
from torch.nn.functional import one_hot as OH
import random


class DataModule:
    """
    Datamodule with sampler for training, inference and creation of a sgb
    """
    def __init__(self, hparam: dict, flag: str, path_inf_data: str = None):
        """
        initializes the datamodule by setting a seed and loading the hparams into the object.
        if used in training mode creates subsets of the data and labels and loads them into object.
        if used in inference mode creates one filtered dataframe and loads it into object.

        Params:
            hparam: dictionary of current experiment
            flag: signals if training(train) mode or inference(inference) mode
            path_inf_data: path to csv used for inference, only needed if flag="inference" ist set
        """
        torch.manual_seed(hparam["MAN_SEED"])
        torch.cuda.manual_seed(hparam["MAN_SEED"])
        random.seed(hparam["MAN_SEED"])
        super().__init__()
        torch.manual_seed(hparam["MAN_SEED"])
        torch.cuda.manual_seed(hparam["MAN_SEED"])
        random.seed(hparam["MAN_SEED"])
        self.hparam = hparam

        if flag == "sgb":
            self.fil_df = pd.read_csv(path_inf_data)

        if flag == "train":
            self.x_train, self.x_val, self.x_test, self.label_train, self.label_val, self.label_test = self.sampler(self.hparam["DATA_DIR"])

        if flag == "inference":
            fil_df = pd.read_csv(path_inf_data)
            self.x_inf, self.label_inf = self.inference_sampler(fil_df)

    def sampler(self, list_of_df):
        """
        sampler for training
        cuts out overlapping windows from each column of one dataframe and assigns integer label to each column.
        Adds both to two separate but corresponding lists.
        creates sub lists of these lists for training, validation and testing

        params:
            list_of_df: list of paths to csv datasets

        Returns:
            x_train: data used for training
            x_val: data used for validation
            x_test: data used for testing
            label_train: corresponding labels for training
            label_val: corresponding labels for validation
            label_test: corresponding labels for testing
        """
        df_list = []
        label_list = []
        # iterate over all dataframes
        for k in list_of_df:
            z = 0
            full_data = pd.read_csv(k)
            #full_data = full_data.drop(
            #    ["actual_joint_voltage_0", "actual_joint_voltage_1", "actual_joint_voltage_2", "actual_joint_voltage_3",
            #     "actual_joint_voltage_4", "actual_joint_voltage_5", "actual_main_voltage","actual_robot_voltage","actual_robot_current",], axis=1)
            # cut out overlapping window from each column and assign each column unique integer label
            for m in full_data.columns:
                # scale each complete column
                data = minmax_scale(full_data[m], feature_range=(-1, 1))
                #data = full_data[m]
                interval = self.hparam["SAMPLE_SIZE"]
                for l in range(0, len(data), math.ceil(interval / 6)):
                    data1 = data[l:l + interval]
                    # check if cutout has specified length
                    if (len(data1)) == interval:
                        df_list.append(data1)
                        label_list.append(z)

                z = z + 1
        x_train, x_val, label_train, label_val = train_test_split(df_list, label_list, train_size=0.5, shuffle=True)
        x_val, x_test, label_val, label_test = train_test_split(x_val, label_val, test_size=0.5, shuffle=False)
        label_train, label_val, label_test = OH(torch.tensor(label_train), num_classes=-1), OH(torch.tensor(label_val), num_classes=-1), OH(torch.tensor(label_test), num_classes=-1)

        return x_train, x_val, x_test, label_train, label_val, label_test

    def inference_sampler(self, df):
        """
        sampler for inference
        cuts out overlapping windows from each column of one dataframe and assigns integer label to each column.
        Adds both to two separate but corresponding lists.
        creates sub lists of these lists for training, validation and testing

        Params:
            df: dataframe of data for inference

        Returns:
            x_inf: list of dataframes/samples
            label_inf: one hot encoded tensor with corresponding labels
        """
        x_inf = []
        label_list = []
        z = 0

        # cut out overlapping window from each column and assign each column unique integer label
        for m in df.columns:
            # scale each complete column
            data = minmax_scale(df[m].astype(float), feature_range=(-1, 1))
            interval = self.hparam["SAMPLE_SIZE"]
            for l in range(0, len(data), math.ceil(interval/6)):
                data1 = data[l:l + interval]
                # check if cutout has specified length
                if (len(data1)) == interval:
                    x_inf.append(data1)
                    label_list.append(z)

            z = z + 1
        label_inf = OH(torch.tensor(label_list), num_classes=-1)
        return x_inf, label_inf

    def sgb_sampler(self, fil_df):
        """
        sampler for sgb creation
        cuts out overlapping windows from inference data and checks if they have the same length
        saves both the column label and the cutout data in separate but corresponding lists

        Params:
            fil_df: dataframe with data for inference

        Returns:
            df_list: list of cutout columns
            label_list: list of labels for cutouts
        """
        df_list = []
        label_list = []
        z = 0
        for j in fil_df.columns:
            data = minmax_scale(fil_df[j].astype(float), feature_range=(-1, 1))
            interval = self.hparam["SAMPLE_SIZE"]
            for k in range(0, len(data), math.ceil(interval/6)):
                data1 = data[k:k + interval]
                if (len(data1) == interval):
                    df_list.append(data1)
                    label_list.append(z)
            z = z + 1
        return df_list, label_list

    def train_loader(self):
        """
        creates a TensorDataset from training data(x_train) and corresponding labels(label_train)

        Returns:
            train_dataloader: dataloader for training
        """
        x_train = np.asarray(self.x_train)
        #train_ds = TensorDataset(torch.tensor(x_train, dtype=torch.float), torch.tensor(self.label_train))
        train_ds = TensorDataset(torch.tensor(x_train, dtype=torch.float), self.label_train)
        train_dataloader = DataLoader(train_ds, batch_size=self.hparam["BATCH_SIZE"], num_workers=self.hparam["NUM_WORKERS"],
                          shuffle=True)
        return train_dataloader

    def val_loader(self):
        """
        creates a TensorDataset from validation data(x_val) and corresponding labels(label_val)

        Returns:
            val_dataloader: dataloader for validation
        """
        x_val = np.asarray(self.x_val)
        val_ds = TensorDataset(torch.tensor(x_val, dtype=torch.float), self.label_val)
        val_dataloader = DataLoader(val_ds, batch_size=self.hparam["BATCH_SIZE"], num_workers=self.hparam["NUM_WORKERS"],
                          shuffle=False)
        return val_dataloader

    def test_loader(self):
        """
        creates a TensorDataset from testing data(x_test) and corresponding labels(label_test)

        Returns:
            test_dataloader: dataloader for testing
        """
        x_test = np.asarray(self.x_test)
        test_ds = TensorDataset(torch.tensor(x_test, dtype=torch.float), self.label_test)
        test_dataloader = DataLoader(test_ds, batch_size=self.hparam["BATCH_SIZE"], num_workers=self.hparam["NUM_WORKERS"],
                          shuffle=False)
        return test_dataloader

    def sgb_loader(self):
        """
        creates a TensorDataset from inference data(x_inf) and corresponding labels(label_inf)

        Returns:
            inference_dataloader: dataloader for inference
        """
        x_sgb, label_sgb = self.sgb_sampler(self.fil_df)
        sgb_ds = TensorDataset(torch.tensor(np.asarray(x_sgb), dtype=torch.float), torch.tensor(label_sgb))
        sgb_dataloader = DataLoader(sgb_ds, batch_size=self.hparam["BATCH_SIZE"], num_workers=self.hparam["NUM_WORKERS"], shuffle=False)
        return sgb_dataloader

    def inference_loader(self):
        x_inf = np.asarray(self.x_inf)
        inf_ds = TensorDataset(torch.tensor(x_inf, dtype=torch.float), self.label_inf)
        inference_dataloader = DataLoader(inf_ds, batch_size=self.hparam["BATCH_SIZE"],
                                     num_workers=self.hparam["NUM_WORKERS"],
                                     shuffle=False)
        return inference_dataloader


if __name__ == "__main__":
    exp_setup = {
        "EXPERIMENT_ID": 0,
        "MODEL": "Test_network",
        "DATA_DIR": [r"I:\BA\Code\datasets\ur10_var_1.csv", r"I:\BA\Code\datasets\ur10_var_2.csv"],
        "LOG_DIR": r"C:\Users\Alexander\Desktop\BA\Code\logs/",
        "MAN_SEED": 42,
        "SAMPLE_SIZE": 1000,
        "BATCH_SIZE": 1024,
        "NUM_WORKERS": 4,
        "MAX_EPOCHS": 3,
        "LEARNING_RATE": 0.0001,
        "WEIGHT_DECAY": 1e-05,
        "CONV_N_FILT": 1,
        "CONV_ENCS": [1, 2, 2, 4, 4],
        "CONV_N_KERNEL": 8,
        "CONV_STRIDE": 1,
        "CONV_DIL": 1,
        "VAR_N_LAT": 4,
        "CORE_ENCS": [1.0, 0.5, 0.25, 0.05],
        "KL_WEIGHT": 0.0001
    }

    data = DataModule(hparam=exp_setup, flag="sgb", path_inf_data=r"I:/BA/Code/datasets/ur10_var_1.csv")

    train_data = data.sgb_loader()

    # print first sample in train_dataloader
    X, y = [x for x in iter(train_data).next()]

    print("X shape: ", X.shape)
    print("y shape: ", y.shape)

    for i, (x, y) in enumerate(train_data):
        print(x.shape)