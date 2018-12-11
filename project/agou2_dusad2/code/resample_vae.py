import numpy as np
import torch

class VAEResampler:
    def __init__(self, dataset):
        self.dataset = dataset
        self.full_data = None
        self.load_full_dataset()

    def load_full_dataset(self):
        # takes a dataset and loads it entirely into memory
        self.full_data = torch.stack([self.dataset[i] for i in range(len(self.dataset))])

    def get_peaks(self):
        # data is the full dataset loaded in memory
        
        # preprocess dataset to inverted grayscale
        norm_dataset = (1-(self.full_data.sum(1)/3)).numpy()
        # get peaks of pixel distributions
        norm_px_dist = norm_dataset.sum(1)
        peaks = np.argmax(norm_px_dist, axis=1)
        return peaks

    def get_sample_weights(self):
        # data is the full dataset loaded in memory

        # get peaks of pixel distributions (where the car is)
        peaks = self.get_peaks()

        #counts of peaks at x coordinates
        counts = np.bincount(peaks)

        # we invert the counts, normalize, and take softmax so low counts have high prob and high counts have low prob
        # norm_inv_counts = (-counts + counts.mean())/(counts.max() - counts.mean())
        # samp_prob = np.exp(norm_inv_counts)/np.exp(norm_inv_counts).sum())

        # replace each sample's peak with the count class of that peak
        counts_by_peaks = counts[peaks]
        # same as above except instead of a sample prob across counts
        # it's a sample prob across samples weighted by its count class
        norm_inv_counts_by_peaks = (-counts_by_peaks + counts_by_peaks.mean())/(counts_by_peaks.max() - counts_by_peaks.mean())
        samp_prob_by_peaks = np.exp(norm_inv_counts_by_peaks)/np.exp(norm_inv_counts_by_peaks).sum()

        return samp_prob_by_peaks


