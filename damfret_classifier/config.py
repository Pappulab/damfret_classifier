import numpy as np
from copy import copy


# This number is determined by analyzing the Shannon Entropy of the datasets.
NUMBER_OF_BINS = 300

DEFAULT_LOW_CONC = 0.0
DEFAULT_HIGH_CONC = 5.0


class Config(object):
    def __init__(self, number_of_bins_xy=NUMBER_OF_BINS):
        low_conc_cutoff = 1.5  # data lower than this value will be excluded
        high_conc_cutoff = 5.0  # data higher than this value will be excluded

        low_conc = 0.0
        high_conc = 5.0
        low_fret = -1.0
        high_fret = 1.0
        conc_bin_width = 0.2    # This effectively creates 25 bins within the current range
        fret_bin_width = 0.02   # This effectively creates 100 bins within the current range
        num_conc_bins = int((high_conc - low_conc)/conc_bin_width) + 1
        num_fret_bins = int((high_fret - low_fret)/fret_bin_width) + 1

        # These are the limits of the data for the cutoff. This data is excluded 
        # as data the extrema are noisy. Notably, the upper limit is 5.0. 
        lower_conc = 1.5
        higher_conc = copy(high_conc)

        conc_bins = np.linspace(low_conc, high_conc, num_conc_bins)
        fret_bins = np.linspace(low_fret, high_fret, num_fret_bins)
        fg_conc_bins = np.linspace(low_conc, high_conc, number_of_bins_xy + 1)
        fg_fret_bins = np.linspace(low_fret, high_fret, number_of_bins_xy + 1)

        num_replicates = -1  # set at runtime


