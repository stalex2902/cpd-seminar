"""Baseline seq2seq methods."""

class BaselineDataset(Dataset):
    """Class for datasets corresponded to extra baselines (simple, weak_labels)."""

    def __init__(
        self,
        cpd_dataset: Dataset,
        baseline_type: str = "simple",
        subseq_len: int = None,
    ) -> None:
        """Initialize baseline dataset.

        :param cpd_dataset: original CPD dataset used for baseline dataset generation
        :param baseline_type: type of evaluated baseline method: simple or weak_labels
            * Simple - we consider each vector from sequence independently.
            So, from one CPD object [x_1, x_2, ..., x_N] and labels [y_1, y_2, ..., y_N],
            we get N elements (x_i, y_i).
            * Weak_labels - we consider subsequences from initial sequence with one label.
            So, from one CPD object [x_1, x_2, ..., x_N], labels [y_1, y_2, ..., y_N] and
            subseq_len = W, we get N - W + 1 objects of the form ([x_i, ..., x_{i+W}], y_{i+W})
        :param subseq_len: length of subsequence for weak_labels
        """
        def _get_subset_(
            dataset: Dataset,
            subset_size: int,
            shuffle: bool = True,
            random_seed: int = 123,
        ) -> Dataset:
            """Get subset of dataset.

            :param dataset: dataset
            :param subset_size: desired size of subset
            :param shuffle: if True shuffle elements, default True
            :param random_seed: seed for experiments reproducibility default 123
            :return: subset of dataset
            """
            fix_seeds(random_seed)

            len_dataset = len(dataset)
            idx = np.arange(len_dataset)

            if shuffle:
                idx = random.sample(list(idx), min(subset_size, len_dataset))
            else:
                idx = idx[:subset_size]
            subset = Subset(dataset, idx)
            return subset

        self.baseline_type = baseline_type
        self.cpd_dataset = cpd_dataset
        self.dataset_len = len(cpd_dataset)

        # TODO: change to seq_len?
        # get length of one sequence
        self.seq_len = len(cpd_dataset.__getitem__(0)[1])

        self.subseq_len = subseq_len

        if (self.baseline_type == "weak_labels") and (self.subseq_len is None):
            raise ValueError("Please, set subsequence length.")

    def __len__(self) -> int:
        """Get length of dataset."""
        if self.baseline_type == "simple":
            return self.dataset_len * self.seq_len
        elif self.baseline_type == "weak_labels":
            return self.dataset_len * (self.seq_len - self.subseq_len + 1)
        else:
            raise ValueError("Wrong type of baseline.")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get one element and corresponded labels.

        :param idx: index of element in dataset
        :return:
            for "simple" - one vector from sequence and one corresponded label
            for "weak_labels" - subsequences and one corresponded label
        """
        sequence, labels = None, None
        if self.baseline_type == "simple":
            global_idx = idx // self.seq_len
            local_idx = idx % self.seq_len
            sequence = self.cpd_dataset[global_idx][0]
            sequence = sequence[:, local_idx]
            labels = self.cpd_dataset[global_idx][1][local_idx]

        elif self.baseline_type == "weak_labels":
            global_idx = idx // (self.seq_len - self.subseq_len + 1)
            local_idx = idx % (self.seq_len - self.subseq_len + 1)
            sequence = self.cpd_dataset[global_idx][0][
                :, local_idx : local_idx + self.subseq_len
            ]
            labels = self.cpd_dataset[global_idx][1][
                :, local_idx : local_idx + self.subseq_len
            ][-1]

        return sequence, labels


# Extra baseline methods.
class ZeroBaseline(nn.Module):
    """Class for Zero baseline."""
    def __init__(self) -> None:
        """Initialize Zero baseline."""
        super().__init__()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Get zero predictions.

        It is the most primitive baseline (no change points).

        :param inputs: input data
        :return: tensor with all zeros
        """
        batch_size, seq_len = inputs.size()[:2]
        out = torch.zeros((batch_size, seq_len, 1))
        return out


# TODO: think how to do it by torch calculations
class L2Baseline(nn.Module):
    """Class for CPD based on L2 distance."""
    def __init__(self, l2_type: str,
                 extractor: Optional[nn.Module] = None,
                 device: str = 'cuda') -> None:
        """Initialise L2 Baseline.

        :param l2_type: type of L2 calculation:
            - one_by_one - ||x_i - x_{i - 1}||
            - vs_first - ||x_i - x_1||
            - vs_mean - ||x_i - \hat{x}||
        :param extractor: feature extractor, e.g. resnet, default None
        :param device: device for calculation, default cuda
        """
        super().__init__()
        self.device = device
        self.type = l2_type
        self.extractor = extractor

        if self.type not in ["one_by_one", "vs_first", "vs_mean"]:
            raise ValueError("Wrong type name {}.".format(self.type))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Calculate L2 distance between sequence elements.

        It is an alternative to provide probability. After the calculation, we should compare
        obtained distances with the threshold.

        :param inputs: input data
        :return: tensor with L2 distance
        """
        batch_size, seq_len = inputs.size()[:2]
        l2_dist = []

        if self.extractor is not None:
            inputs = extractor(inputs)
        for seq in inputs:
            seq = seq.float().to(self.device)
            if self.type == "one_by_one":
                curr_l2_dist = [0] + [((x - y) ** 2).sum().item() for x, y in zip(seq[1:], seq[:-1])]
            elif self.type == "vs_first":
                curr_l2_dist = [0] + [((x - seq[0]) ** 2).sum().item() for x in seq[1:]]
            elif self.type == "vs_mean":
                mean_seq = torch.mean(seq, 0)
                curr_l2_dist = [0] + [((x - mean_seq) ** 2).sum().item() for x in seq[1:]]
            curr_l2_dist = np.array(curr_l2_dist) / max(curr_l2_dist)
            l2_dist.append(curr_l2_dist)
        #l2_dist = torch.from_numpy(np.array(l2_dist))
        # TODO: check
        l2_dist = torch.stack(l2_dist)
        return l2_dist