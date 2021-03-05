import os
import re
import time
import shutil
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.optimize import curve_fit
from collections import OrderedDict, namedtuple
from tabulate import tabulate

from damfret_classifier.config import Config
from damfret_classifier.logger import setup_logger
from damfret_classifier.parameters import Parameters
from damfret_classifier.plotting import plot_gaussian_fits, plot_logistic_fits
from damfret_classifier.plotting import plot_linear_rsquared_fit, plot_fine_grid_profiles
from damfret_classifier.utils import load_settings, load_raw_synthetic_data, load_manuscript_classifications
from damfret_classifier.utils import read_original_data, remove_genes_and_replicates_below_count
from damfret_classifier.utils import apply_cutoff_to_dataframe, to_fwf, determine_if_manioc_project
from damfret_classifier.utils import create_genes_table, clamp_data, check_if_data_within_limits
from damfret_classifier.utils import initialize_pool, parallelize, determine_well_name


__all__ = ['clamp_data', 'calculate_rsquared', 'slice_data', 'calculate_nucleated_fractions', 'determine_class',
           'classify_dataset', 'classify_datasets']


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

    residuals   = ydata - func(xdata, *popt)
    ss_res      = np.sum(residuals**2.0)
    ss_tot      = np.sum((ydata - np.mean(ydata))**2.0)
    r_squared   = 1.0 - (ss_res / ss_tot)

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
    fit_bounds  = [[0.001, 0.001, 0.001, 0.001], [1.0, 1.0, 1.0, 1.0]]
    gauss2      = lambda x, a1, c1, a2, c2: a1*np.exp(-((x)/c1)**2.0) + a2*np.exp(-((x-average)/c2)**2.0)

    slice_number        = 1
    nucleated_fractions = list()
    conc_bin_centers    = list()
    r_squared           = list()

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
            plot_params['savepath']       = config.work_dir
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
        score   = 1.0  # originally: score = (0.15-diff)/0.15
        color   = 'blue'

    elif fraction_above_csat < 0.1:
        score1  = min([1,(diff-0.15)/(0.5-0.15)])
        score2  = (0.1-fraction_above_csat)/0.1
        score   = min([score1, score2])
        color   = 'yellow'

    elif fit_value > 0.08 and np.min(region_r_squared) < 0.6:
        score1  = min([1, (0.6 - np.min(region_r_squared))/(0.6 - 0.3)])
        score2  = min([1,(diff-0.15)/(0.5-0.15)])
        score3  = min([1,(fraction_above_csat-0.1)/(0.3-0.1)])
        score   = min([score1, score2, score3])
        color   = 'red'

    elif r > 0.6:
        score1  = (np.min(region_r_squared)-0.6)/(1-0.6)
        score2  = min([1,(diff-0.15)/(0.5-0.15)])
        score3  = (r-0.6)/(1-0.6)
        score4  = min([1,(fraction_above_csat-0.1)/(0.3-0.1)])
        score   = min([score1, score2, score3, score4])
        color   = 'magenta'

    else:
        score1  = (np.min(region_r_squared)-0.6)/(1-0.6) # score 3
        score2  = min([1,(diff-0.15)/(0.5-0.15)])  # score 1
        score3  = (0.6-r)/(0.6) # score 4
        score4  = min([1,(fraction_above_csat-0.1)/(0.3-0.1)])  # score 2
        score   = min([score1, score2, score3, score4])
        color   = 'green'
    return color, score


# ---------------------------------------------------------------------------------------------------------------------
# LOGGING FORMATTING HELPERS (PRIVATE TO THIS MODULE ONLY)


def _insert_section_separator(logger):
    """This is a helper function to output a section separator within a given logger instance.

    @param logger (`logging.Logger`): An already initialized `logging.Logger()` instance which
                                      will be written to. Since logs are global and can be
                                      accessed from anywhere, this method of access is allowed.
    @return None
    """
    logger.info('{}'.format('=' * 80))


def _log_conc_FRET_averages(logger, average, conc_fret_averages):
    """This function outputs the concentration FRET averages into the provided logger instance.
    This approach is preferred as it shortens the overall classification function.

    @param logger (`logging.Logger`): An already initialized `logging.Logger()` instance which
                                      will be written to. Since logs are global and can be
                                      accessed from anywhere, this method of access is allowed.
    @param average (float):           The average FRET concentration.
    @param conc_fret_averages (list): The list containing the FRET averages for each
                                      concentration slice.
    """
    logger.info('Conc FRET average: {}'.format(average))
    logger.info('All conc FRET averages: {}'.format(conc_fret_averages))
    logger.info('Last 4 conc FRET averages: {}'.format(conc_fret_averages[-4:]))


def _log_region_values(logger, region_indices, region_lower, region_upper, conc_bin_centers):
    """This is a convenience function for logging some of the more esoteric parameters calculated
    during the analysis. These are primarily for debugging purposes."""
    logger.info('Region Indices: {}'.format(region_indices))
    logger.info('Region lower: {}'.format(region_lower))
    logger.info('Region upper: {}'.format(region_upper))
    logger.info('Concentration bin centers at region lower: {}'.format(conc_bin_centers[region_lower]))
    logger.info('Concentration bin centers at region upper: {}'.format(conc_bin_centers[region_upper]))
    logger.info('Concentration bin centers using region indices: {}'.format(conc_bin_centers[region_indices]))


def _log_class_and_score(logger, well_file, color, score):
    """Output the classification and its confidence score for a given well into the provided log object."""
    message1 = 'Analysis :: Well file: "{}" class: "{}" score: "{}"'.format(well_file, color, score)
    message2 = 'Processing complete for well file "{}"'.format(well_file)
    logger.info(message1)
    logger.info(message2)


def _log_skipped_analysis(logger, well_file, low_conc, high_conc, min_conc, max_conc):
    """Indicate to the user why the provided well file was skipped for analysis due to it having
    data that exceeded the provided concentration range. Such well files are saved to `skipped.{TSV,CSV}`."""
    message1 = 'Skipped :: well file "{}" exceeds the defined concentration limits.'.format(well_file)
    message2 = 'Input concentration limits: ({}, {})'.format(low_conc, high_conc)
    message3 = 'Actual concentration limits: ({}, {})'.format(min_conc, max_conc)
    message4 = 'This file will not be analyzed and marked with zeroes & N/A in the final parameter file.'
    message5 = 'Processing complete for well file "{}"'.format(well_file)
    logger.info(message1)
    logger.info(message2)
    logger.info(message3)
    logger.info(message4)
    logger.info(message5)



def _configure_logger(logs_dir, session_name, raw_well_name, attach_stream_handler):
    """This is somewhat important as it sets up the logger for use during multithreading. The end
    result of this is that each individual well file has their own log."""
    well_log_filepath = os.path.join(logs_dir, '{well}.log'.format(well=raw_well_name))
    log_name = '{}_{}'.format(session_name, raw_well_name)
    setup_logger(log_name, well_log_filepath, attach_stream_handler)
    logger = logging.getLogger(log_name)
    return logger


def _log_skipped_file_not_found(logger, well_file):
    """Notify the user that the well file was not found, and hence was skipped."""
    message1 = 'Skipped :: well file "{}" not found.'.format(well_file)
    message2 = 'This well will be marked with zeroes & N/A in the final parameter file.'
    message3 = 'Processing complete for well file "{}"'.format(well_file)
    logger.info(message1)
    logger.info(message2)
    logger.info(message3)


def _log_skipped_cell_measurements(logger, well_file, required_counts, counts):
    """Indicate to the user why the input well file was skipped due to the number of cell measurements
    not matching the required minimum number, as indicated by via the config file."""
    message1 = 'Skipped :: well file "{}" is less than the required minimum number of cell measurements.'.format(well_file)
    message2 = 'Input minimum required cell measurements: {}'.format(required_counts)
    message3 = 'Actual cell measurements: {}'.format(counts)
    message4 = 'This file will not be analyzed and marked with zeroes & N/A in the final parameter file.'
    message5 = 'Processing complete for well file "{}"'.format(well_file)
    logger.info(message1)
    logger.info(message2)
    logger.info(message3)
    logger.info(message4)
    logger.info(message5)


def _plot_skipped_well_file(config, raw_data, well_name):
    """Determine whether or not to generate a plot of the fine grid profile of the well
    which was skipped. This is useful when the user wants to visually interrogate the
    profile and determine appropriate limits for a subsequent rerun."""
    if config.plot_skipped:
        plot_params = dict()
        plot_params['savepath']         = config.work_dir
        plot_params['well_name']        = well_name
        plot_params['fg_conc_edges']    = config.fg_conc_edges
        plot_params['fg_fret_edges']    = config.fg_fret_edges
        plot_params['data_df']          = raw_data
        plot_params['fine_grid_xlim']   = None
        plot_params['fine_grid_ylim']   = None
        plot_params['plot_type']        = config.plot_type
        plot_fine_grid_profiles(plot_params)


def _write_session_log_header(config):
    """It's important for the user to know what settings have been used throughout this analysis.
    This outputs those parameters, and saves them to the current session log."""
    session_log = logging.getLogger(config.session_name)
    wells_table = pd.read_csv(config.wells_filename)

    _insert_section_separator(session_log)
    session_log.info('')
    session_log.info('Session started on: {}'.format(datetime.now()))
    session_log.info('Number of wells to be analyzed: {}'.format(len(wells_table['well_file'])))
    session_log.info('')
    _insert_section_separator(session_log)
    session_log.info('')
    session_log.info('SETTINGS OVERVIEW')
    session_log.info('')

    # First, output all values of the important variables:
    settings = load_settings(config.settings_filename)
    for setting_name in settings:
        value = settings[setting_name]
        auto = ''
        if setting_name == 'random_seed' and value is None:
            value = getattr(config, 'random_seed')
            auto = '(auto-populated)'
        elif setting_name == 'num_processes' and value is None:
            value = getattr(config, 'num_processes')
            auto = '(auto-populated)'
        elif setting_name == 'minimum_required_measurements':
            value = getattr(config, 'min_measurements')
        elif setting_name == 'work_directory' and value is None:
            value = getattr(config, 'work_dir')
            auto = '(auto-populated)'
        elif setting_name == 'logs_directory' and value is None:
            value = getattr(config, 'logs_dir')
            auto = ' (auto-populated)'

        v = value
        if type(value) is str:
            v = '"{}"'.format(value)

        session_log.info('{setting}: {value} {gen_method}'.format(setting=setting_name, value=v, gen_method=auto))
    session_log.info('')
    _insert_section_separator(session_log)
    session_log.info('')


def _initialize_session_log(config):
    """Initial the log and dump the header ahead of analysis."""
    session_log = logging.getLogger(config.session_name)
    _write_session_log_header(config)

    session_log.info('BEGIN ANALYSIS')
    session_log.info('')
    return session_log


def _insert_analysis_summary_section(session_log):
    """Add the analysis summary section to the session log. Mainly for beautification."""
    session_log.info('')
    session_log.info('ANALYSIS COMPLETE')
    session_log.info('')
    _insert_section_separator(session_log)
    session_log.info('')
    session_log.info('SUMMARY')
    session_log.info('')


def _save_parameters_and_skipped_wells(config, session_log, wells_table, results):
    """After the analysis is complete, examine the parameters and skipped well files
    (if any), and save them as TSVs and CSVs. The parameters file is large and
    contains all the columns corresponding to the `Parameters` object, although
    not all columns are populated. The `skipped` files are truncated versions of the
    input `wells_filename` as defined in the config."""

    # check for which wells were skipped.
    parameter_fields = (
        'gene construct replicate well_file counts mean_fret gauss2_loc ' \
        'csat csat_slope linear_r2 max_gauss_r2 min_r2_region max_r2_region ' \
        'min_abs_diff_r2_region max_abs_diff_r2_region frac_above_csat color score'
    )
    fields = parameter_fields.split()

    # Prepare the `pandas.DataFrame` for use.
    parameters = OrderedDict()
    for field in fields:
        parameters[field] = list()

    skipped = list()
    for fargs in results:
        raw_well_file = fargs[1]
        params, _start_time, _stop_time = results[fargs]
        if params.counts == 0 and params.score == 0 and params.color == 'N/A':
            skipped.append(raw_well_file)

        for field in fields:
            col = getattr(params, field)
            parameters[field].append(col)

    skipped_dfs = list()
    for skipped_well in skipped:
        df = wells_table[wells_table['well_file'] == skipped_well].dropna()
        skipped_dfs.append(df)

    if len(skipped_dfs) > 0:
        _save_skipped_wells(config, session_log, skipped_dfs)
    _save_parameters(config, session_log, parameters)
    return parameters, skipped


def _save_skipped_wells(config, session_log, skipped_dfs):
    """Export the skipped wells, and indicate as much in the session log."""
    skipped_df = pd.concat(skipped_dfs)
    skipped_df.reset_index(drop=True, inplace=True)
    skipped_savepath_tsv = os.path.join(config.work_dir, 'skipped.tsv')
    skipped_savepath_csv = os.path.join(config.work_dir, 'skipped.csv')

    to_fwf(skipped_df, skipped_savepath_tsv)
    skipped_df.to_csv(skipped_savepath_csv, index=False)
    session_log.info('Skipped well files exported as: "{}"'.format(skipped_savepath_tsv))
    session_log.info('Skipped well files exported as: "{}"'.format(skipped_savepath_csv))


def _save_parameters(config, session_log, parameters):
    """Export the parameters, and indicate as much in the session log."""
    parameters_df           = pd.DataFrame(parameters)
    parameters_savepath_tsv = os.path.join(config.work_dir, 'parameters.tsv')
    parameters_savepath_csv = os.path.join(config.work_dir, 'parameters.csv')

    to_fwf(parameters_df, parameters_savepath_tsv)
    parameters_df.to_csv(parameters_savepath_csv, index=False)
    session_log.info('Classifications and scores exported as: "{}"'.format(parameters_savepath_tsv))
    session_log.info('Classifications and scores exported as: "{}"'.format(parameters_savepath_csv))
    session_log.info('')


def _output_summary(session_log, wells_table, skipped, start_time):
    """Summarize and output the analysis performed."""
    total_wells         = len(wells_table['well_file'])
    skipped_wells       = len(skipped)
    analyzed_wells      = total_wells - skipped_wells
    stop_time           = datetime.now()
    processing_time     = stop_time - start_time

    session_log.info('')
    session_log.info('Number of wells to analyze:       {}'.format(total_wells))
    session_log.info('Number of wells analyzed:         {}'.format(analyzed_wells))
    session_log.info('Number of wells skipped:          {}'.format(skipped_wells))
    session_log.info('Processing time completed on:     {}'.format(stop_time))
    session_log.info('Total processing time:            {}'.format(processing_time))
    session_log.info('')
    _insert_section_separator(session_log)


def _move_plots_to_project_root(config, session_log):
    """Move all plots from the `work` directory to the `project_directory`. This always
    occurs with MANIOC projects. For other projects, this function has to be called
    explicitly."""
    session_log.info('')
    _insert_section_separator(session_log)
    session_log.info('')
    session_log.info('MOVE PLOTS')
    session_log.info('')
    session_log.info('Moving files from `work_directory` ("{}") to `project_directory` ({})'.format(
        config.work_dir,
        config.project_dir
    ))
    plot_files = [f for f in os.listdir(config.work_dir) if f.endswith(config.plot_type)]
    for fname in plot_files:
        original_path = os.path.join(config.work_dir, fname)
        path_for_copy = os.path.join(config.project_dir, fname)
        shutil.move(original_path, path_for_copy)
        session_log.info('Moved: "{}" to "{}"'.format(original_path, path_for_copy))
    session_log.info('')
    session_log.info('MOVE COMPLETE')
    session_log.info('')
    _insert_section_separator(session_log)


# ---------------------------------------------------------------------------------------------------------------------
# CLASSIFICATION FUNCTIONS


def classify_dataset(config, raw_well_name):
    """This function that does the primary work in classifying the dataset. It is specifically
    designed for multithreaded use in that each function call is indepedent of the other.

    @param config (Config):         A parsed and populated `Config` object specific to the
                                    project / dataset.

    @param raw_well_name (str):     The raw name of the well (i.e. zero-padded) which has
                                    to be parsed in other to determine the non-zero-padded
                                    filename that will be loaded and subsequently analyzed.

    @return params (Parameter):     A `Parameters` object which records what features were
                                    calculated.

    Once all the well files are analyzed in this way, the `Parameters` objects can then be
    collated and saved.
    """
    np.random.seed(config.random_seed)
    params = Parameters()

    logistic_func = lambda x, b, c: 1.0/(1.0+np.exp(-(x-b)/c))
    logistic_bounds = [[1, 0], [20, 2]]
    p0 = (5, 0.3)

    # Set the nice / priority level for the function when it
    # is executed. This becomes important when multiprocessing is applied.
    os.nice(config.nice_level)

    # We use the `raw_well_name` - e.g. A01 which has the leading zero.
    # The purpose of this is that when the log is saved, it can easily
    # be sorted and collated into the final session log.
    logger = _configure_logger(config.logs_dir, config.session_name, raw_well_name, attach_stream_handler=True)
    session = logging.getLogger(config.session_name)

    # Determine the actual well name
    well_name = determine_well_name(raw_well_name)
    well_file = os.path.join(config.project_dir, config.filename_format.format(well_name=well_name))

    # Begin processing.
    message = 'Processing well file "{}"'.format(well_file)
    session.info(message)
    logger.info(message)

    # First, check if this file exists
    if not os.path.exists(well_file):
        _log_skipped_file_not_found(session, well_file)
        _log_skipped_file_not_found(logger, well_file)
        params.well_file = well_name
        return params

    # Next, check to determine if this file has enough cell measurements for analysis (as limited by the
    # Shannon entropy) due to the input grid size.
    raw_data, raw_counts = read_original_data(well_file, config.low_conc_cutoff, config.high_conc_cutoff, apply_cutoff=False)
    if raw_counts < config.min_measurements:
        _plot_skipped_well_file(config, raw_data, well_name)
        _log_skipped_cell_measurements(session, well_file, config.min_measurements, raw_counts)
        _log_skipped_cell_measurements(logger, well_file, config.min_measurements, raw_counts)
        params.well_file = well_name
        return params

    # Next, check if the file is within the data limits:
    valid, min_conc, max_conc = check_if_data_within_limits(raw_data, config.low_conc_cutoff, config.high_conc_cutoff)
    if not valid:
        _plot_skipped_well_file(config, raw_data, well_name)
        _log_skipped_analysis(session, well_file, config.low_conc, config.high_conc, min_conc, max_conc)
        _log_skipped_analysis(logger, well_file, config.low_conc, config.high_conc, min_conc, max_conc)
        params.well_file = well_name
        return params

    # At this point, we can continue
    data, counts        = apply_cutoff_to_dataframe(raw_data, config.low_conc_cutoff, config.high_conc_cutoff)
    params.counts       = counts
    params.well_file    = well_name

    # remove extreme concentration values
    data = clamp_data(data, config.low_conc_cutoff, config.high_conc_cutoff)
    params.mean_fret = data['damfret'].mean()

    # plot the fine-grid plots
    if config.plot_fine_grids:
        plot_params = dict()
        plot_params['savepath']         = config.work_dir
        plot_params['well_name']        = well_name
        plot_params['fg_conc_edges']    = config.fg_conc_edges
        plot_params['fg_fret_edges']    = config.fg_fret_edges
        plot_params['data_df']          = data
        plot_params['fine_grid_xlim']   = config.fine_grid_xlim
        plot_params['fine_grid_ylim']   = config.fine_grid_ylim
        plot_params['plot_type']        = config.plot_type
        plot_fine_grid_profiles(plot_params)

    # Calculate the concentration FRET averages and the max of these averages
    # This will then be used to characterize nucleation by setting the max average
    # as the center for a Gaussian, and setting another Gaussian at 0. A double
    # Gaussian function will be fit to the 1D histogram of the FRET for each concentration
    # slice, and the quality of that fit will serve as a proxy for nucleation.
    conc_fret_averages, slices_histograms = slice_data(data, config.conc_bins, config.fret_bins)
    average = max(conc_fret_averages[-4:])  # previously from [-4:]
    _log_conc_FRET_averages(logger, average, conc_fret_averages)

    # Determine how the system nucleates across concentration by calculating the area under the
    # Gaussian whose center is set to the max average. Then, that area is compared to the total
    # area under both Gaussians. Systems with specific nucleation profiles have unique
    # nucleation patterns.
    params.gauss2_loc = average
    nucleated_fractions, conc_bin_centers, r_squared = calculate_nucleated_fractions(config, well_name, slices_histograms, average)

    # We also save the quality of fit of the double Gaussians.
    # Since the location of the second Gaussian (i.e. the max of the concentration averages) is
    # fixed, the quality of fit across the concentration slices is an additional proxy for
    # a continuous transition.
    conc_bin_centers    = np.array(conc_bin_centers)
    r_squared           = np.array(r_squared)
    params.max_gauss_r2 = np.max(r_squared)

    # fit a linear function to the R^2 values of the Gauss2 fits across the different slices.
    linear_func         = lambda x, a, b: a*x + b
    popt, pconv         = curve_fit(linear_func, conc_bin_centers, r_squared, bounds=[[-np.inf, -np.inf], [0, np.inf]])
    linear_func_data    = linear_func(conc_bin_centers, *popt)
    _p, linear_rsquared = calculate_rsquared(linear_func, conc_bin_centers, r_squared, bounds=[[-np.inf, -np.inf], [0, np.inf]])

    if config.plot_rsquared:
        plot_params = dict()
        plot_params['savepath']         = config.work_dir
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
        mean_fret       = data['damfret'].mean()
        score           = (0.15-diff)/0.15
        params.score    = score

        if mean_fret < 0.05:
            params.color = 'blue'
        else:
            params.color = 'black'
        _log_class_and_score(session, well_file, params.color, params.score)
        _log_class_and_score(logger, well_file, params.color, params.score)
        return params

    # Now, fit a logistic function to the data
    popt, logistic_rs = calculate_rsquared(logistic_func, conc_bin_centers, nucleated_fractions, p0, logistic_bounds)
    saturation_conc, slope = popt
    logistic_y = logistic_func(conc_bin_centers, *popt)
    if config.plot_logistic:
        plot_params = dict()
        plot_params['savepath']             = config.work_dir
        plot_params['well_name']            = well_name
        plot_params['conc_bin_centers']     = conc_bin_centers
        plot_params['nucleated_fractions']  = nucleated_fractions
        plot_params['logistic_y']           = logistic_y
        plot_params['r_squared']            = logistic_rs
        plot_params['popt']                 = popt
        plot_params['plot_type']            = config.plot_type
        plot_logistic_fits(plot_params)

    # This is the final check for black.
    if saturation_conc <= config.low_conc_cutoff:
        score = (0.15-diff)/0.15

        params.csat         = saturation_conc
        params.csat_slope   = slope
        params.color        = 'black'
        params.score        = 1.0

        _log_class_and_score(session, well_file, params.color, params.score)
        _log_class_and_score(logger, well_file, params.color, params.score)
        return params

    # Record the saturation concentration.
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
    _log_region_values(logger, region_indices, region_lower, region_upper, conc_bin_centers)

    # Determine if there is a large change in the fit goodness in this region. And, determine the max change in
    # fit goodness around the saturation concentration.
    region_r_squared = r_squared[region_indices]
    logger.info('Region R^2: {}'.format(region_r_squared))
    fit_value = np.max(np.abs(np.diff(region_r_squared)))

    params.csat                     = saturation_conc
    params.csat_slope               = slope
    params.linear_r2                = linear_rsquared
    params.min_r2_region            = np.min(region_r_squared)
    params.max_r2_region            = np.max(region_r_squared)
    params.min_abs_diff_r2_region   = np.min(np.abs(np.diff(region_r_squared)))
    params.max_abs_diff_r2_region   = np.max(np.abs(np.diff(region_r_squared)))

    # check for noisy data that hasn't fully phase separated
    upper_conc_limit = config.high_conc_cutoff - 1
    if saturation_conc <= upper_conc_limit:
        df = data[data['concentration'] > saturation_conc].dropna()
    else:
        df = data[data['concentration'] > upper_conc_limit].dropna()
    df = df[df['damfret'] > 0.05].dropna()

    total_points            = len(data)
    df_points               = len(df)
    fraction_above_csat     = df_points/total_points
    params.frac_above_csat  = fraction_above_csat

    # Finally, determine the phase separation class and its confidence score based on the
    # various parameters calculated.
    color, score = determine_class(df_points, fraction_above_csat, diff, fit_value, region_r_squared, linear_rsquared)
    params.color = color
    params.score = score

    # Log and quit.
    _log_class_and_score(session, well_file, color, score)
    _log_class_and_score(logger, well_file, color, score)
    return params


def classify_datasets(config_filename, move_to_project_root=False):
    """This is the main entry point for the classification of the datasets using
    a provided configuration file. This is so configured such that the user can
    use the entry point for both MANIOC and non-MANIOC project directories.

    @param config_filename (str):       The location of the YAML config file which
                                        will be used when analyzing the data.

    @param move_to_project_root (bool): Whether or not to move the plots from
                                        the `work_dir` to the `project_directory`.
                                        This is only required for MANIOC project
                                        layouts. However, it can be requested by
                                        the user if so inclined.

    @return None

    When this function completes, the analysis (`parameters.{tsv,csv}`) is saved
    to a `work` directory saved under the `project_directory` (if no `work` directory
    is provided). A session log is also output to that location, in addition to a
    `skipped.{tsv,csv}` file which is a truncated version of `wells_filename` that
    contains all the wells which were skipped during the analysis. All plots are
    either stored in the `work` directory by default, or in the `project_directory`
    if requested by setting `move_to_project_root=True` when calling this function.
    """
    start_time  = datetime.now()
    config      = Config(config_filename)

    # Setup the logger for use.
    iso_time                = start_time.strftime('%Y-%m-%d_%H-%M-%S')
    session_log_filename    = os.path.join(config.work_dir, 'session---%s.log' % iso_time)
    setup_logger(config.session_name, session_log_filename, attach_stream_handler=True)
    session_log = _initialize_session_log(config)
    wells_table = pd.read_csv(config.wells_filename)

    # Calculate the classification and score for each well.
    function_args = list()
    for _index, row in wells_table.iterrows():
        raw_well_name = str(row['well_file'])
        args = (config, raw_well_name, )
        function_args.append(args)

    # Save the results / parameters as an `OrderedDict`
    results = parallelize(classify_dataset, function_args, config.num_processes, config.session_name)

    if move_to_project_root:
        _move_plots_to_project_root(config, session_log)

    # Reformat and save the parameters and skipped wells (if any).
    _insert_analysis_summary_section(session_log)
    _parameters, skipped = _save_parameters_and_skipped_wells(config, session_log, wells_table, results)
    _insert_section_separator(session_log)
    _output_summary(session_log, wells_table, skipped, start_time)

    # Shutdown and quit the current session.
    logging.shutdown()
    del session_log
