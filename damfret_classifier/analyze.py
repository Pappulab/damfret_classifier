import os
import time
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.optimize import curve_fit
from collections import OrderedDict
from tabulate import tabulate
from damfret_classifier.plotting import plot_gaussian_fits, plot_logistic_fits, plot_linear_rsquared_fit, plot_fine_grid_profiles
from damfret_classifier.utils import load_raw_synthetic_data, load_manuscript_classifications, create_directory_if_not_exist
from damfret_classifier.utils import load_settings, read_original_data, remove_genes_and_replicates_below_count
from damfret_classifier.utils import create_genes_table, clamp_data
from damfret_classifier.config import Config
from damfret_classifier.logger import logging as logger


__all__ = 'clamp_data,calculate_rsquared,slice_data,calculate_nucleated_fractions,determine_class,classify_datasets'.split(',')


# Supremely useful answer to convert a DataFrame to a TSV file: https://stackoverflow.com/a/35974742/866930
def to_fwf(df, fname):
    """This is a convenience function which is used for the easier generation of TSV files from panda.DataFrame
    objects. It is meant to be tacked on as a function to a Pandas Dataframe object. Since this effectively
    monkey-patches the code, it has to be performed on a per file basis."""
    content = tabulate(df.values.tolist(), list(df.columns), tablefmt="plain", floatfmt='.5f')


# ---------------------------------------------------------------------------------------------------------------------


def calculate_rsquared(func, xdata, ydata, p0=None, bounds=None, maxfev=10000):
    """This is a convenience function which encapsulates the calculation of the `r_squared`
    value of a function fit to passed `xdata` and `ydata`.

    @param func:   The function which will be used when fitting and the x and y data.
                   Note that a `lambda` function is preferable here as subsequent calls
                   to named functions will reuse previous arguments, a quirk.
    @param xdata:  The x data of the function which will be fitted.
    @param ydata:  The y data of the function which will be fitted.
    @param p0:     A tuple or list containing the initial values which will be used when
                   evaluating the function.  
    @param bounds: A tuple or list containing the upper and lower bounds of the parameters
                   passed to the fitted function, `func`. For e.g. if the input function
                   requires 2 parameters, the list or tuple passed to `bounds` will contain
                   2 items, each a list or tuple of size 2.
    @param maxfev: An integer referring to the number of times the function should be
                   evaluated. 

    @return tuple: A 2-tuple comprised of the `popt` and `rsquared` from a fit of the
                   function data based on the passed parameters.
    """
    if p0 is not None and bounds is None:
        popt, _pcov = curve_fit(func, xdata, ydata, p0=p0, maxfev=maxfev)
    
    elif p0 is not None and bounds is not None:
        popt, _pcov = curve_fit(func, xdata, ydata, p0=p0, bounds=bounds, maxfev=maxfev)

    if p0 is None and bounds is None:
        popt, _pcov = curve_fit(func, xdata, ydata, maxfev=maxfev)
    elif p0 is None and bounds is not None:
        popt, _pcov = curve_fit(func, xdata, ydata, bounds=bounds, maxfev=maxfev)

    residuals = ydata - func(xdata, *popt)
    ss_res = np.sum(residuals**2.0)
    ss_tot = np.sum((ydata - np.mean(ydata))**2.0)
    r_squared = 1.0 - (ss_res / ss_tot)

    return popt, r_squared


def slice_data(data, conc_bins, fret_bins):
    """This function takes the input `data` and edges `conc_bins` & `fret_bins`
    and creates slices of the FRET data within those limits. These slices are
    histogrammed, and then checked to determine their nucleated state, which
    is done using the fitting of Gauss2 functions as a proxy for nucleation.

    @param data (pandas.DataFrame): a DataFrame object holding the raw concentration
                                    and FRET data.
    @param conc_bins (np.array):    a 1D numpy array containing the edges along the
                                    concentration axis.

    @param fret_bins (np.array):    a 1D numpy array containing the edges along the
                                    FRET axis.
    
    @return tuple:                  a 2-tuple containing the averages of the FRET
                                    within each concentration slice (i.e. between
                                    two concentration edges); and, an OrderedDict
                                    containing the concentration slices as keys
                                    and another 2-tuple as values. That 2-tuple
                                    contains the selected data between those limits,
                                    and a sum-normalized 1D histogram of the FRET
                                    data corresponding to the FRET bins.

    Note: the edges are wider than the edges used for the generation of fine-grid 
    histogram plots.
    """
    # now, go through the concentration slices
    conc_fret_averages = list()
    slices_histograms = OrderedDict()
    for low, high in zip(conc_bins[:-1], conc_bins[1:]):
        df = data.copy()
        df = df[df['concentration'] > low]
        df = df[df['concentration'] <= high]
        
        # histogram the fret data in the concencration slice
        if not df.empty:
            fret_hist, _edges = np.histogram(df['damfret'], bins=fret_bins)
            norm_fret_hist = fret_hist / fret_hist.sum()

            df = df[df['damfret'] > 0.05]
            if not df.empty:
                average_fret = df['damfret'].mean()
                conc_fret_averages.append(average_fret)

                key = (low, high)
                slices_histograms[key] = (df, norm_fret_hist)  # we'll use the norm hist for fitting
    return conc_fret_averages, slices_histograms


def calculate_nucleated_fractions(config, well_name, slices_histograms, average):
    """This function determines the extent to which a given system or dataset has undergone phase
    separation by using the fits of a double Gaussian function (Gauss2) to the 1D histogram of FRET
    data found across a sliced concentration range.

    @param config (Config):                 A Config object corresponding to the analysis parameters
                                            prescribed for the data in question.

    @param well_name (str):                 The name of the well file which will be loaded according
                                            to the config parameters.

    @param slices_histograms (OrderedDict): an OrderedDict containing the concentration slices as keys
                                            and another 2-tuple as values. That 2-tuple contains the 
                                            selected data between those limits, and a sum-normalized 
                                            1D histogram of the FRET data corresponding to the FRET 
                                            bins.

    @param average (float):                 The max average of the FRET within the last 4 slices across
                                            the concentration slices. This utilizes the fact that the
                                            edge of the data at higher concentration tends to taper
                                            off or also increase in FRET.

    @return tuple:                          A 3-tuple corresponding to the `nucleated_fractions`, 
                                            `conc_bin_centers`, and `r_squared` values determined
                                            from the Gauss2 fits.

    The double-Gaussian is configured such that the center of the 2nd Gaussian is always fixed. Hence,
    as the slices are iterated across and the function is fit to the 1D histogram of the FRET data
    within that concentration slice, it can be used as a proxy for nucleation. Nucleation is determined
    by comparing the area of the 2nd Gaussian to the total area of both Gaussians. See the function: 
    `classify_datasets` for more.
    """
    # fit two gaussians to the histogram of FRET values within the bin.
    fit_bounds = [[0.001, 0.001, 0.001, 0.001], [1.0, 1.0, 1.0, 1.0]]
    gauss2  = lambda x, a1, c1, a2, c2: a1*np.exp(-((x)/c1)**2.0) + a2*np.exp(-((x-average)/c2)**2.0)

    slice_number = 1
    nucleated_fractions = list()
    conc_bin_centers = list()
    r_squared = list()

    for conc_slice in slices_histograms:
        _df, norm_fret_hist = slices_histograms[conc_slice]
        x = config.fret_bins[:-1] + config.fret_bin_width/2.0
        
        popt, rs = calculate_rsquared(gauss2, x, norm_fret_hist, bounds=fit_bounds)
        a1, c1, a2, c2 = popt
        
        g1 = lambda x, a, c: a*np.exp(-((x)/c)**2.0)
        g2 = lambda x, a, c: a*np.exp(-((x-average)/c)**2.0)
        y1 = g1(x, a1, c1)
        y2 = g2(x, a2, c2)

        frac_nucleated = np.trapz(y2)/(np.trapz(y1) + np.trapz(y2))
        if config.plot_gaussian:
            plot_params = dict()
            plot_params['savepath']       = config.plots_dir
            plot_params['well_name']      = well_name
            plot_params['norm_fret_hist'] = norm_fret_hist
            plot_params['x']              = x
            plot_params['y1']             = y1
            plot_params['y2']             = y2
            plot_params['average']        = average
            plot_params['conc_slice']     = conc_slice
            plot_params['frac_nucleated'] = frac_nucleated
            plot_params['slice_number']   = slice_number
            plot_params['plot_type']      = config.plot_type
            plot_gaussian_fits(plot_params)

        nucleated_fractions.append(frac_nucleated)
        conc_bin_center = conc_slice[0] + config.conc_bin_width/2.0

        conc_bin_centers.append(conc_bin_center)
        r_squared.append(rs)
        slice_number += 1
    return nucleated_fractions, conc_bin_centers, r_squared


def determine_class(df_points, fraction_above_csat, diff, fit_value, region_r_squared, r):
    """This function determines what phase separated class a given dataset has based on
    the parameters calculated which quantify nucleation. Note, the parameter choices 
    used to characterize a given class is purely phenomenological. 

    @param df_points (int):             The number of points found at a FRET above 0.05.

    @param fraction_above_csat (float): The fraction of points found above the csat of 0.05.

    @param diff (float):                The difference between the last and minimum value of
                                        the nucleated fractions.

    @param fit_value (float):           The max of the absolute difference of the R^2 values
                                        of the R^2 values surrounding the calculated
                                        saturation concentration.

    @param region_r_squared (np.array): The R^2 values selected from the indices closest to
                                        the calculated saturation concentration as determined
                                        from the logistic function fit.

    @param r (float):                   The R^2 value from a linear function fit to the R^2
                                        values of the Guass2 function across concentration
                                        slices.

    @return tuple:                      A 2-tuple corresponding to the color and confidence score
                                        respectively of the dataset. The color has 5 possibilities:
                                        black, blue, red, gree, magenta, and yellow. See below for
                                        a more detailed description. The confidence score ranges
                                        from 0 to 1 with 1 signifying 100% confidence in the
                                        assignment of the accompanying color / class.

    Color meanings:
        black:      Assembled at all concentrations
        blue:       No assembly at all concentrations
        red:        Continuous Transition
        green:      Discontinuous Transition
        magenta:    Higher Order State
        yellow:     Incomplete Transition
    """
    color = None
    score = None
    if df_points <= 20:
        score = 1.0  # originally: score = (0.15-diff)/0.15
        color = 'blue'
    
    elif fraction_above_csat < 0.1:
        score1 = min([1,(diff-0.15)/(0.5-0.15)])
        score2 = (0.1-fraction_above_csat)/0.1
        score = min([score1, score2])
        color = 'yellow'
    
    elif fit_value > 0.08 and np.min(region_r_squared) < 0.6:
        score1 = min([1, (0.6 - np.min(region_r_squared))/(0.6 - 0.3)])
        score2 = min([1,(diff-0.15)/(0.5-0.15)])
        score3 = min([1,(fraction_above_csat-0.1)/(0.3-0.1)])
        score = min([score1, score2, score3])
        color = 'red'

    elif r > 0.6:
        score1 = (np.min(region_r_squared)-0.6)/(1-0.6)
        score2 = min([1,(diff-0.15)/(0.5-0.15)])
        score3 = (r-0.6)/(1-0.6)
        score4 = min([1,(fraction_above_csat-0.1)/(0.3-0.1)])
        score = min([score1, score2, score3, score4])
        color = 'magenta'

    else:
        score1 = (np.min(region_r_squared)-0.6)/(1-0.6) # score 3
        score2 = min([1,(diff-0.15)/(0.5-0.15)])  # score 1
        score3 = (0.6-r)/(0.6) # score 4
        score4 = min([1,(fraction_above_csat-0.1)/(0.3-0.1)])  # score 2
        score = min([score1, score2, score3, score4])
        color = 'green'
    return color, score


def classify_datasets(settings, config, genes_table):
    """This function does the heavy lifting and is the main entry point for processing
    and classifying the data in a single-threaded process.

    @param settings (pandas.DataFrame):     A pandas DataFrame containing the project settings.
    @param config (Config):                 A config object initialized according to parameters
                                            suited to the data being analyzed.
    @param genes_table (pandas.DataFrame):  A pandas DataFrame comprised of all the gene, plasmid,
                                            and well_name data.

    At completion this function generates intermediate plots for debugging as indicated, as well
    as a numpy array containing the confidence scores and class / color assignments. Lastly, it
    also generates a summary file, `parameters.tsv` which contains all the parameters of interest
    to the calculation and class / color determination, which is useful for debugging.
    """
    if not isinstance(config, Config):
        raise RuntimeWarning('A `damfret_classifier.Config` object is required.')
    
    # Set the random seed for reproducibility.
    random_seed = settings['random_seed']
    if random_seed is None:
        random_seed = int(datetime.now().timestamp())
        np.random.seed(random_seed)
    logger.info('Using random seed: {}.'.format(random_seed))

    # A lambda function is used for fitting as the parameters are dynamically set.
    # Typically defined functions i.e. using `def` is not mutable and only one reference
    # is populated - remaining calls are not updated with new values.
    logistic_func = lambda x, b, c: 1.0/(1.0+np.exp(-(x-b)/c))
    logistic_bounds = [[1, 0], [20, 2]]
    p0 = (5, 0.3)

    params = OrderedDict()
    params['gene']                      = list()
    params['construct']                 = list()
    params['replicate']                 = list()
    params['well_file']                 = list()
    params['counts']                    = list()
    params['mean-fret']                 = list()
    params['gauss2-loc']                = list()
    params['csat']                      = list()
    params['csat-slope']                = list()
    params['linear-r2']                 = list()
    params['max-gauss-r2']              = list()
    params['min-r2-region']             = list()
    params['max-r2-region']             = list()
    params['min-abs-diff-r2-region']    = list()
    params['max-abs-diff-r2-region']    = list()
    params['frac-above-csat']           = list()
    params['color']                     = list()
    params['score']                     = list()

    counted = list()
    confidence_scores = OrderedDict()
    logger.info('Beginning classification calculations.')
    replicate_number = 0
    for replicate_index, row in genes_table.iterrows():
        well = int(row.well_file[1:])  # works for `A09` and `A9`
        well_name = '%s%d' % (row.well_file[0], well)  # reformat to match CSV data files name format.
        
        # compare against the actual data
        well_file = os.path.join(config.data_dir, config.filename_format.format(well_name=well_name))
        filename = well_file[:]

        logger.info('Processing filename [{:03d} / {:03d}]: {}'.format(replicate_index + 1, len(genes_table), filename))
        print('Processing filename [{:03d} / {:03d}]: {}'.format(replicate_index + 1, len(genes_table), filename))

        data, counts = read_original_data(well_file, config.low_conc_cutoff, config.high_conc_cutoff)
        counted.append(well_file)
        replicate = len(counted)
        if len(counted) == config.num_replicates:
            counted = list()
        
        params['counts'].append(counts) # NOT `len(data)` as it has already been pruned
        params['gene'].append(row.gene)
        params['well_file'].append(row.well_file)
        
        # remove extreme concentration values
        data = clamp_data(data, config.low_conc_cutoff, config.high_conc_cutoff)
        params['mean-fret'].append(data['damfret'].mean())
        
        # plot the fine-grid plots
        if config.plot_fine_grids:
            plot_params = dict()
            plot_params['savepath']         = config.plots_dir
            plot_params['well_name']        = well_name
            plot_params['fg_conc_edges']    = config.fg_conc_edges
            plot_params['fg_fret_edges']    = config.fg_fret_edges
            plot_params['data_df']          = data
            plot_params['fine_grid_xlim']   = config.fine_grid_xlim
            plot_params['fine_grid_ylim']   = config.fine_grid_ylim
            plot_params['plot_type']        = config.plot_type
            plot_fine_grid_profiles(plot_params)

        conc_fret_averages, slices_histograms = slice_data(data, config.conc_bins, config.fret_bins)
        average = max(conc_fret_averages[-4:])  # previously from [-4:]
        
        logger.info('Conc FRET average: {}'.format(average))
        logger.info('All conc FRET averages: {}'.format(conc_fret_averages))
        logger.info('Last 4 conc FRET averages: {}'.format(conc_fret_averages[-4:]))
        params['gauss2-loc'].append(average)
        nucleated_fractions, conc_bin_centers, r_squared = calculate_nucleated_fractions(config, well_name, slices_histograms, average)

        conc_bin_centers = np.array(conc_bin_centers)
        r_squared = np.array(r_squared)
        params['max-gauss-r2'].append(np.max(r_squared))
        
        # fit a linear function to the R^2 values of the Guass2 fits across the different slices.
        linear_func = lambda x, a, b: a*x + b
        popt, pconv = curve_fit(linear_func, conc_bin_centers, r_squared, bounds=[[-np.inf, -np.inf], [0, np.inf]])
        linear_func_data = linear_func(conc_bin_centers, *popt)
        _p, linear_rsquared = calculate_rsquared(linear_func, conc_bin_centers, r_squared, bounds=[[-np.inf, -np.inf], [0, np.inf]])

        if config.plot_rsquared:
            plot_params = dict()
            plot_params['savepath']         = config.plots_dir
            plot_params['conc_bin_centers'] = conc_bin_centers
            plot_params['r_squared']        = r_squared
            plot_params['linear_func_data'] = linear_func_data
            plot_params['well_name']        = well_name
            plot_params['r']                = linear_rsquared
            plot_params['popt']             = popt
            plot_params['plot_type']        = config.plot_type
            plot_linear_rsquared_fit(plot_params)

        # now, analyze the fraction that's nucleated to determine if the system is in one or two-states
        diff = nucleated_fractions[-1] - np.min(nucleated_fractions)
        if nucleated_fractions[-1] - np.min(nucleated_fractions) < 0.15:
            mean_fret = data['damfret'].mean()
            params['construct'].append(row.construct)
            params['replicate'].append(replicate)
            params['csat'].append(0)
            params['csat-slope'].append(0)
            params['linear-r2'].append(0)
            params['min-r2-region'].append(0)
            params['max-r2-region'].append(0)
            params['min-abs-diff-r2-region'].append(0)
            params['max-abs-diff-r2-region'].append(0)
            params['frac-above-csat'].append(0)
            
            if mean_fret < 0.05:
                score = (0.15-diff)/0.15
                confidence_scores[filename] = ('blue', score)
                
                params['color'].append('blue')
                params['score'].append(score)
            else:
                score = (0.15-diff)/0.15
                confidence_scores[filename] = ('black', score)

                params['color'].append('black')
                params['score'].append(score)
            continue  # short-circuit

        # now, fit a logistic function to the data
        popt, logistic_rs = calculate_rsquared(logistic_func, conc_bin_centers, nucleated_fractions, p0, logistic_bounds)
        saturation_conc, slope = popt
        logistic_y = logistic_func(conc_bin_centers, *popt)
        if config.plot_logistic:
            plot_params = dict()
            plot_params['savepath']             = config.plots_dir
            plot_params['well_name']            = well_name
            plot_params['conc_bin_centers']     = conc_bin_centers
            plot_params['nucleated_fractions']  = nucleated_fractions
            plot_params['logistic_y']           = logistic_y
            plot_params['r_squared']            = logistic_rs
            plot_params['popt']                 = popt
            plot_params['plot_type']            = config.plot_type
            plot_logistic_fits(plot_params)

        if saturation_conc <= config.low_conc_cutoff:
            score = (0.15-diff)/0.15
            confidence_scores[filename] = ('black', 1.00)

            params['construct'].append(row.construct)
            params['replicate'].append(replicate)
            params['csat'].append(saturation_conc)
            params['csat-slope'].append(slope)
            params['linear-r2'].append(0)
            params['min-r2-region'].append(0)
            params['max-r2-region'].append(0)
            params['min-abs-diff-r2-region'].append(0)
            params['max-abs-diff-r2-region'].append(0)
            params['frac-above-csat'].append(0)
            params['color'].append('black')
            params['score'].append(1.0)
            continue
        
        logger.info('Saturation concentration: {}'.format(saturation_conc))
        
        if saturation_conc < max(conc_bin_centers):
            region_indices = np.where((conc_bin_centers >= saturation_conc - 1.0) & (conc_bin_centers <= saturation_conc + 1.0))
            logger.info('Saturation concentration type: lesser')
        else:
            indices = list(range(len(conc_bin_centers)))
            region_indices = indices[-5:]
            logger.info('Saturation concentration type: greater')

        region_lower = np.where(conc_bin_centers >= saturation_conc)
        region_upper = np.where(conc_bin_centers <= saturation_conc)
        
        logger.info('Rregion Indices: {}'.format(region_indices))
        logger.info('Region lower: {}'.format(region_lower))
        logger.info('Region upper: {}'.format(region_upper))
        logger.info('Concentration bin centers at region lower: {}'.format(conc_bin_centers[region_lower]))
        logger.info('Concentration bin centers at region upper: {}'.format(conc_bin_centers[region_upper]))
        logger.info('Concentration bin centers using region indices: {}'.format(conc_bin_centers[region_indices]))
        
        # determine if there is a large change in the fit goodness in this region. And, determine the max change in
        # fit goodness around the saturation concentration.
        region_r_squared = r_squared[region_indices]
        logger.info('Region R^2: {}'.format(region_r_squared))
        fit_value = np.max(np.abs(np.diff(region_r_squared)))
        
        params['construct'].append(row.construct)
        params['replicate'].append(replicate)
        params['csat'].append(saturation_conc)
        params['csat-slope'].append(slope)
        params['linear-r2'].append(linear_rsquared)
        params['min-r2-region'].append(np.min(region_r_squared))
        params['max-r2-region'].append(np.max(region_r_squared))
        params['min-abs-diff-r2-region'].append(np.min(np.abs(np.diff(region_r_squared))))
        params['max-abs-diff-r2-region'].append(np.max(np.abs(np.diff(region_r_squared))))
        
        # check for noisy data that hasn't fully phase separated
        upper_conc_limit = config.high_conc_cutoff - 1
        if saturation_conc <= upper_conc_limit:
            df = data[data['concentration'] > saturation_conc].dropna()
        else:
            df = data[data['concentration'] > upper_conc_limit].dropna()
        df = df[df['damfret'] > 0.05].dropna()

        total_points = len(data)
        df_points = len(df)
        fraction_above_csat = df_points/total_points
        params['frac-above-csat'].append(fraction_above_csat)
        
        color, score = determine_class(df_points, fraction_above_csat, diff, fit_value, region_r_squared, linear_rsquared)
        confidence_scores[filename] = (color, score)
        params['color'].append(color)
        params['score'].append(score)
    logger.info('End classification calculations.')

    confidence_scores_savename = os.path.join(config.work_dir, 'confidence-scores.npy')
    np.save(confidence_scores_savename, confidence_scores, allow_pickle=True)

    params_savename = os.path.join(config.work_dir, 'parameters.tsv')
    params = pd.DataFrame(params)
    params.to_fwf(params_savename)

    time.sleep(10)
