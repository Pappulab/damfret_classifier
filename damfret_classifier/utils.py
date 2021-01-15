import io
import pkgutil
import yaml
import numpy as np
import pandas as pd
import multiprocessing as mp
from datetime import datetime
from collections import OrderedDict
from scipy.optimize import curve_fit


__all__ = ['load_raw_synthetic_data', 'load_config', 'read_original_data', 'calculate_rsquared', 
           'start_process', 'initialize_pool', 'parallelize']


# ---------------------------------------------------------------------------------------------------------------------


def load_raw_synthetic_data():
    """This function loads the raw synthetic data referenced in the DAmFRET manuscript.

    @return synthetic_data: an `OrderedDict` containing the raw synthetic data. 
                            Keys are short names. Values are 2-tuples. The first
                            index contains a `Pandas.DataFrame` with 2 columns 
                            ("concentration", "FRET"). Some of the rows are 
                            populated with NaNs. The second index contains the 
                            corresponding long filename.

    Useful reference: https://programtalk.com/vs2/python/12103/python-skyfield/skyfield/functions.py/
    """
    synthetic_data_bytes = pkgutil.get_data('damfret_classifier', 'synthetic.npy')
    synthetic_data = np.load(io.BytesIO(synthetic_data_bytes), allow_pickle=True).item()
    return synthetic_data


# ---------------------------------------------------------------------------------------------------------------------


def load_config(config_filename):
    """This function loads a YAML configuration file into a dictionary object.

    @param config_filename: The filename of the YAML config file to load.
    @return dict: a dictionary populated as key / value pairs from the YAML config file.
    """
    with open(config_filename) as cfile:
        return yaml.load(cfile)


def read_original_data(filename, low_conc_cutoff, high_conc_cutoff):
    data = pd.read_csv(filename, usecols=['Acceptor-A', 'FRET-A'])
    counts = len(data)  # Record the original number of points
    
    df = pd.DataFrame()
    df['concentration'] = np.log10(data['Acceptor-A'])
    df['damfret'] = data['FRET-A']/data['Acceptor-A']

    # Remove extraneous data not found between the limits.
    df = df[df['concentration'] >= low_conc_cutoff]
    df = df[df['concentration'] <= high_conc_cutoff]
    df = df.dropna()  # Remove any rows with NaN values in the FRET columns.
    df.reset_index(drop=True, inplace=True)  # Renumber the indices of the rows kept for simplicity.

    return df, counts


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


# ---------------------------------------------------------------------------------------------------------------------


def start_process():
    """This is just a notifier function to indicate that a worker process has been launched. 
    Since this is meant for a pool, the pool reuses these processes."""
    print('Starting', mp.current_process().name)


def initialize_pool(num_processes, initializer_function):
    """Initialize a multiprocessing pool based on the number of requested processes and
    some initializer function which is used primarily for notification."""
    num_procs = num_processes
    if num_processes is None:
        num_procs = mp.cpu_count()
    
    pool = mp.Pool(processes=num_procs, initializer=initializer_function)
    return pool


def parallelize(func, list_of_func_args, num_processes=None):
    """This function parallelizes calling a function with different argument values.

    @param func (func):                 The function reference which will be called.
    @param num_repeats (int):           The number of times to call the function.
    @param func_args (tuple | list):    The arguments to pass to the function call.
    @param num_processes (int):         The number of processes to use (default = all).

    @returns results (OrderedDict):     A dictionary of function arguments as keys, and
    a values of 2-tuples comprised of datetime instances and the function result. Note:
    the result is a lazy reference to an `ApplyAsync` result. To extract the values, 
    they can be obtained by calling `.get()` on the result.

    Notes: calling the `ApplyAsync` object's get (`result.get()`) here will block, which
    defeats the entire purpose behind this configuration.
    """
    pool = initialize_pool(num_processes, start_process)
    
    results = OrderedDict()
    for func_args in list_of_func_args:
        result = pool.apply_async(func, args=func_args)
        current_time = datetime.utcnow()
        func_result = (result, current_time)
        results[func_args] = func_result
    
    pool.close()
    pool.join()
    return results