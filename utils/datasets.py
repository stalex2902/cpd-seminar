"""Module for experiments' dataset methods."""
from typing import List, Tuple

import random

import numpy as np
import pandas as pd
import torch

from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils.data import Dataset, Subset

def fix_seeds(seed: int) -> None:
    """Fix random seeds for experiments reproducibility.

    :param seed: random seed
    """
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

class CPDDatasets:
    """Class for experiments' datasets."""

    def __init__(
        self,
        experiments_name: str,
        random_seed: int = 123,
        train_anomaly_num: int = None,
    ) -> None:
        """Initialize dataset class.

        :param experiments_name: type of experiments
            Available now:
            - "synthetic_kD" (k is the vector dimensions)
            - "human_activity"
        :param random_seed: seed for reproducibility, default = 123
        """
        super().__init__()
        self.random_seed = random_seed
        self.train_anomaly_num = train_anomaly_num

        if experiments_name == "human_activity":
            self.experiments_name = experiments_name
        elif experiments_name.startswith("synthetic"):
            # format is "synthetic_nD"
            self.dim = int(experiments_name.split("_")[1].split("D")[0])
            self.experiments_name = experiments_name
        else:
            raise ValueError("Wrong experiment_name {}.".format(experiments_name))

    def get_dataset_(self) -> Tuple[Dataset, Dataset]:
        """Load experiments' dataset follow the paper's settings.

        See https://dl.acm.org/doi/abs/10.1145/3503161.3548182
        """
        if self.experiments_name.startswith("synthetic"):
            dataset = SyntheticNormalDataset(
                seq_len=128, num=1000, dim=self.dim, random_seed=self.random_seed
            )
            train_dataset, test_dataset = self.train_test_split_(
                dataset, test_size=0.3, shuffle=True
            )

        else:
            path_to_data = "../data/human_activity/"
            train_dataset = HumanActivityDataset(
                path_to_data=path_to_data, seq_len=20, train_flag=True
            )
            test_dataset = HumanActivityDataset(
                path_to_data=path_to_data, seq_len=20, train_flag=False
            )
        return train_dataset, test_dataset

    def train_test_split_(
        self, dataset: Dataset, test_size: float = 0.3, shuffle: bool = True
    ) -> Tuple[Dataset, Dataset]:
        """Split dataset on train and test.

        :param dataset: dataset for splitting
        :param test_size: size of test data, default 0.3
        :param shuffle: if True - shuffle data, default True
        :return: tuple of
            - train dataset
            - test dataset
        """
        # fix seeds
        fix_seeds(self.random_seed)

        len_dataset = len(dataset)
        idx = np.arange(len_dataset)

        # train-test split
        if shuffle:
            train_idx = random.sample(list(idx), int((1 - test_size) * len_dataset))
        else:
            train_idx = idx[: -int(test_size * len_dataset)]
        test_idx = np.setdiff1d(idx, train_idx)

        train_set = Subset(dataset, train_idx)
        test_set = Subset(dataset, test_idx)
        return train_set, test_set


class SyntheticNormalDataset(Dataset):
    """Class for Dataset of synthetically generated time series."""

    def __init__(
        self, seq_len: int, num: int, dim: int = 1, random_seed: object = 123
    ) -> None:
        """Initialize dataset.

        :param seq_len: length of generated time series
        :param num: number of object in dataset
        :param dim: dimension of each vector in time series (default 1, 1D)
        :param random_seed: seed for results reproducibility (default 123)
        """
        super().__init__()

        self.seq_len = seq_len
        self.num = num
        self.dim = dim
        self.random_seed = random_seed

        self.data, self.labels = self.generate_synthetic_data()

    def __len__(self) -> int:
        """Get datasets' length.

        :return: length of dataset
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[np.array, np.array]:
        """Get one time series and corresponded labels.

        :param idx: index of element in dataset
        :return: tuple of
             - time series
             - sequence of labels
        """
        return self.data[idx], self.labels[idx]

    def generate_synthetic_data(
        self, multi_dist: bool = False
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Generate synthetic data with change and without change.

        :param multi_dist: if True, the first part mean change randomly, if False - it's always 1
        :return: list of
            - generated time series
            - corresponded labels
        """
        fix_seeds(self.random_seed)

        # get set of random change points
        idxs_changes = torch.randint(1, self.seq_len, (self.num // 2,))

        data = []
        labels = []

        # generate data with change
        for idx in idxs_changes:
            mean = torch.randint(1, 100, (2,))

            while mean[0] == mean[1]:
                mean = torch.randint(1, 100, (2,))

            if not multi_dist:
                mean[0] = 1

            # generate data before change
            dist = MultivariateNormal(
                mean[0] * torch.ones(self.dim), torch.eye(self.dim)
            )
            first_part = []
            for _ in range(self.seq_len):
                vector = dist.sample()
                first_part.append(vector)
            first_part = torch.stack(first_part)

            # generate data after change
            dist = MultivariateNormal(
                mean[1] * torch.ones(self.dim), torch.eye(self.dim)
            )
            second_part = []
            for _ in range(self.seq_len):
                vector = dist.sample()
                second_part.append(vector)
            second_part = torch.stack(second_part)

            # stack all together
            time_series = torch.cat([first_part[:idx], second_part[idx:]])
            label = torch.cat([torch.zeros(idx), torch.ones(self.seq_len - idx)])
            data.append(time_series)
            labels.append(label)

        # generate data without change
        for idx in range(0, self.num - len(idxs_changes)):
            dist = MultivariateNormal(torch.ones(self.dim), torch.eye(self.dim))
            time_series = []
            for _ in range(self.seq_len):
                vector = dist.sample()
                time_series.append(vector)
            time_series = torch.stack(time_series)
            label = torch.zeros(self.seq_len)

            data.append(time_series)
            labels.append(label)
        return data, labels


class HumanActivityDataset(Dataset):
    """Class for Dataset of HAR time series."""

    def __init__(
        self, path_to_data: str, seq_len: int = 20, train_flag: bool = True
    ) -> None:
        """Initialize HAR dataset.

        :param path_to_data: path to data with HAR datasets (train and test)
        :param seq_len: length of considered time series
        :param train_flag: if True - load train dataset (files are separated), default True
        """
        super().__init__()

        self.path_to_data = path_to_data
        self.seq_len = seq_len
        self.train_flag = train_flag

        self.data = self._load_data()
        normal_data, normal_labels = self._generate_normal_data()
        anomaly_data, anomaly_labels = self._generate_anomaly_data()

        self.features = normal_data + anomaly_data
        self.labels = normal_labels + anomaly_labels

    def _load_data(self) -> pd.DataFrame:
        """Load HAR dataset.

        :return: dataframe with HAR data
        """
        if self.train_flag:
            type_ = "train"
        else:
            type_ = "test"

        total_path = self.path_to_data + "/T" + type_[1:]
        data_path = total_path + "/X_" + type_ + ".txt"
        labels_path = total_path + "/y_" + type_ + ".txt"
        subjects_path = total_path + "/subject_id_" + type_ + ".txt"

        data = pd.read_csv(data_path, sep=" ", header=None)
        names = pd.read_csv(self.path_to_data + "/features.txt", header=None)
        data.columns = [x.replace(" ", "") for x in names[0].values]

        labels = pd.read_csv(labels_path, sep=" ", header=None)
        subjects = pd.read_csv(subjects_path, sep=" ", header=None)

        data["subject"] = subjects
        data["labels"] = labels

        return data

    def _generate_normal_data(self) -> Tuple[List[float], List[int]]:
        """Get normal sequences from data.

        :return: tuple of
            - sequences with time series
            - sequences with labels
        """
        slices = []
        labels = []

        for sub in self.data["subject"].unique():
            tmp = self.data[self.data.subject == sub]
            # labels 7 - 12 characterize the change points
            tmp = tmp[~tmp["labels"].isin([7, 8, 9, 10, 11, 12])]
            normal_ends_idxs = np.where(np.diff(tmp["labels"].values) != 0)[0]

            start_idx = 0

            for i in range(0, len(normal_ends_idxs) - 1):
                # get data before change
                end_idx = normal_ends_idxs[i] + 1
                slice_data = tmp.iloc[start_idx:end_idx]

                # if slice len is enough, generate data
                if len(slice_data) > self.seq_len:
                    for j in range(0, len(slice_data) - self.seq_len):
                        slices.append(slice_data[j : j + self.seq_len])
                        labels.append(np.zeros(self.seq_len))
                start_idx = end_idx
        return slices, labels

    def _generate_anomaly_data(self):
        slices = []
        labels = []

        for sub in self.data["subject"].unique():

            tmp = self.data[self.data.subject == sub]
            # labels 7 - 12 characterize the change points
            tmp_change_only = tmp[tmp.labels.isin([7, 8, 9, 10, 11, 12])]
            # find change indexes
            change_idxs = np.where(np.diff(tmp_change_only["labels"].values) != 0)[0]
            change_idxs = [tmp_change_only.index[0]] + list(
                tmp_change_only.index[change_idxs + 1]
            )
            change_idxs = [-1] + change_idxs

            for i in range(1, len(change_idxs) - 1):
                curr_change = change_idxs[i]

                # find subsequences with change with maximum length
                start_idx = max(change_idxs[i - 1] + 1, curr_change - self.seq_len)
                end_idx = min(change_idxs[i + 1] - 1, curr_change + self.seq_len)
                slice_data = tmp.loc[start_idx:end_idx]

                curr_change = list(slice_data.index).index(curr_change)

                # set labels: 0 - before change point, 1 - after
                slice_labels = np.zeros(len(slice_data))
                slice_labels[curr_change:] = np.ones(len(slice_data) - curr_change)

                if len(slice_data) > self.seq_len:
                    for j in range(0, len(slice_data) - self.seq_len):
                        slices.append(slice_data.iloc[j : j + self.seq_len])
                        labels.append(slice_labels[j : j + self.seq_len])
        return slices, labels

    def __len__(self) -> int:
        """Get datasets' length.

        :return: length of dataset
        """
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[np.array, np.array]:
        """Get one time series and corresponded labels.

        :param idx: index of element in dataset
        :return: tuple of
             - time series
             - sequence of labels
        """
        # TODO: fix hardcode
        sel_features = [
            "tBodyAcc-Mean-1",
            "tBodyAcc-Mean-2",
            "tBodyAcc-Mean-3",
            "tGravityAcc-Mean-1",
            "tGravityAcc-Mean-2",
            "tGravityAcc-Mean-3",
            "tBodyAccJerk-Mean-1",
            "tBodyAccJerk-Mean-2",
            "tBodyAccJerk-Mean-3",
            "tBodyGyro-Mean-1",
            "tBodyGyro-Mean-2",
            "tBodyGyro-Mean-3",
            "tBodyGyroJerk-Mean-1",
            "tBodyGyroJerk-Mean-2",
            "tBodyGyroJerk-Mean-3",
            "tBodyAccMag-Mean-1",
            "tGravityAccMag-Mean-1",
            "tBodyAccJerkMag-Mean-1",
            "tBodyGyroMag-Mean-1",
            "tBodyGyroJerkMag-Mean-1",
            "fBodyAcc-Mean-1",
            "fBodyAcc-Mean-2",
            "fBodyAcc-Mean-3",
            "fBodyGyro-Mean-1",
            "fBodyGyro-Mean-2",
            "fBodyGyro-Mean-3",
            "fBodyAccMag-Mean-1",
            "fBodyAccJerkMag-Mean-1",
            "fBodyGyroMag-Mean-1",
            "fBodyGyroJerkMag-Mean-1",
        ]
        return self.features[idx][sel_features].iloc[:, :-2].values, self.labels[idx]