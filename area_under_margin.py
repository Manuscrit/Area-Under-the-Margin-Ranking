import random

import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset


class ConstantTargetDataset(Dataset):
    """
    Dataset returning a given target for all or only the choosen indexes of the given dataset
    """

    def __init__(self, dataset, target=False, idx_filter=[]):
        self.dataset = dataset
        self.target = target
        self.counter = 0
        self.idx_filter = idx_filter

    def __getitem__(self, item):
        tile, target = self.dataset[item]
        if item in self.idx_filter:
            return tile, self.target
        else:
            return tile, target

    def __len__(self):
        return len(self.dataset)


class AreaUnderTheMarginRanking():
    """
    Implementation of the paper Identifying Mislabeled Data using the Area Under the Margin Ranking: https://arxiv.org/pdf/2001.10528v2.pdf

    Currently the used dataset must not shuffle between epochs !
    """

    # TODO use matrix operations
    # TODO manage the case of dataset with shuffling between epochs
    # TODO Setup full process of filtering a dataset

    def __init__(self):
        # hist_delta_AUM_current_epoch dimensions: [n_sample, 2 (from in_logit & max(out_logits))]
        self.hist_delta_AUM_current_epoch = torch.zeros(size=(0, 2))
        # hist_delta_AUM dimensions: [n_epoch, n_sample, in_logit & max(out_logits)]
        self.hist_delta_AUM = torch.zeros(size=(0, 0, 2))
        self.reference_sample_idx = []

    def accumulate(self, batch_logits, batch_ids, batch_targets):
        """ To be called after batch prediction"""
        for img_logit, img_id, img_target in zip(batch_logits.split([1] * len(batch_logits), dim=0),
                                                 batch_ids.split([1] * len(batch_logits), dim=0),
                                                 batch_targets.split([1] * len(batch_logits), dim=0)):
            img_logit = img_logit.squeeze(dim=0)
            target_logit = img_logit[img_target]
            if img_target < len(img_logit) - 1:
                notarget_logits = torch.cat([img_logit[:img_target], img_logit[img_target + 1:]], dim=0)
            else:
                notarget_logits = img_logit[:img_target]
            notarget_logits = notarget_logits.max()
            self.hist_delta_AUM_current_epoch = torch.cat(
                [self.hist_delta_AUM_current_epoch, torch.tensor([[target_logit, notarget_logits]])], dim=0)

    def accumulate_epoch(self):
        """ To be called at the end of each epoch"""
        if len(self.hist_delta_AUM) == 0:
            self.hist_delta_AUM = self.hist_delta_AUM_current_epoch.unsqueeze(dim=0)
        else:
            self.hist_delta_AUM = torch.cat([self.hist_delta_AUM, self.hist_delta_AUM_current_epoch.unsqueeze(dim=0)],
                                            dim=0)
        self.hist_delta_AUM_current_epoch = torch.zeros(size=(0, 2))

    def get_reference_aum_threshold(self, percentile=0.99):
        reference_aum = self.hist_delta_AUM[:, self.reference_sample_idx, :]  # => [n_epoch, n_sample, in_logit & max(
        # out_logits)]
        reference_aum = torch.tensor(reference_aum)
        reference_aum = reference_aum[..., 0] - reference_aum[..., 1]  # => [n_epoch, n_sample]
        reference_aum = reference_aum.mean(dim=0)
        reference_aum, _ = reference_aum.sort(dim=0, descending=False)  # => [n_sample]
        aum_threshold_at_percentile = reference_aum[int(len(reference_aum) * percentile)]
        return aum_threshold_at_percentile

    def add_reference_class_to_ds(self, dataset, n_class, exclusion_idx=None):
        """
        Will modify a given dataset by adding the reference class with only mislabeled data.
        Original targets of the dataset must be [0...n_class-1].
        Currently dataset must not shuffle between epochs !

        :param dataset: Dataset to be modified
        :param n_class: Original number of class in Dataset (original targets must be [0...n_class-1])
        :param exclusion_idx: Do not add these idx in the reference class
        :return: new dataset with the added reference class
        """
        if exclusion_idx is None:
            exclusion_idx = self.reference_sample_idx
        n_reference_sample = int(len(dataset) / (n_class + 1))
        print("select n_reference_sample", n_reference_sample)
        self.reference_sample_idx = random.sample(population=range(len(dataset)), k=n_reference_sample)

        new = []
        for el in self.reference_sample_idx:
            if el in exclusion_idx or el in new:
                done = False
                new_el = el
                while not done:
                    new_el += 1
                    if new_el not in new:
                        if new_el < len(dataset):
                            done = True
                            new.append(el)
                        else:
                            new_el = 0
            else:
                new.append(el)
        assert len(set(new)) == len(new)

        dataset = ConstantTargetDataset(dataset,
                                        target=(n_class - 1) + 1,
                                        idx_filter=self.reference_sample_idx)
        return dataset, self.reference_sample_idx

    def get_mislabeled(self):
        """

        :return: list(mislabel indexes of image in dataset)
        """
        threshold = self.get_reference_aum_threshold()

        # reference_aum = self.hist_delta_AUM[:,self.reference_sample_idx,:]
        # reference_aum = torch.tensor(reference_aum)
        reference_aum = torch.tensor(self.hist_delta_AUM)
        reference_aum = reference_aum[..., 0] - reference_aum[..., 1]
        reference_aum = torch.tensor(reference_aum).mean(dim=0)
        # print("reference_aum.shape",reference_aum.shape)
        mislabeled = (reference_aum < threshold).nonzero().tolist()
        # print("mislabeled",mislabeled)
        mislabeled = [el for el in mislabeled if el not in self.reference_sample_idx]
        # print("mislabeled",mislabeled)
        print("percentage of mislabeled", len(mislabeled) / (len(reference_aum) - len(self.reference_sample_idx)),
              len(mislabeled), len(reference_aum), len(self.reference_sample_idx))
        return mislabeled

    def get_hist(self):
        """
        To be use to visualize the results. After the training or after each epoch
        """
        sample_1 = random.randint(0, self.hist_delta_AUM.shape[1])
        sample_2 = random.randint(0, self.hist_delta_AUM.shape[1])
        sample_3 = random.randint(0, self.hist_delta_AUM.shape[1])
        sample_4 = random.randint(0, self.hist_delta_AUM.shape[1])
        sample_5 = random.randint(0, self.hist_delta_AUM.shape[1])

        sample_1 = self.hist_delta_AUM[:, sample_1, :]
        sample_2 = self.hist_delta_AUM[:, sample_2, :]
        sample_3 = self.hist_delta_AUM[:, sample_3, :]
        sample_4 = self.hist_delta_AUM[:, sample_4, :]
        sample_5 = self.hist_delta_AUM[:, sample_5, :]

        plt.figure()
        plt.subplot(511)
        plt.plot(sample_1)
        plt.subplot(512)
        plt.plot(sample_2)
        plt.subplot(513)
        plt.plot(sample_3)
        plt.subplot(514)
        plt.plot(sample_4)
        plt.subplot(515)
        plt.plot(sample_5)
        plt.show()
