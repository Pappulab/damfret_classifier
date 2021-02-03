import os
import io
import pkgutil
import yaml
import logging
import numpy as np
import pandas as pd
import multiprocessing as mp
from pathlib import Path
from tabulate import tabulate
from datetime import datetime
from collections import OrderedDict
from scipy.optimize import curve_fit


__all__ = ['load_raw_synthetic_data', 'load_manuscript_classifications', 'create_directory_if_not_exist', 'shannon_entropy',
           'load_settings', 'read_original_data', 'remove_genes_and_replicates_below_count', 'validate_gene_replicates',
           'create_genes_table', 'start_process', 'initialize_pool', 'parallelize', 'generate_default_config', 'to_fwf']


# ---------------------------------------------------------------------------------------------------------------------


# Supremely useful answer to convert a DataFrame to a TSV file: https://stackoverflow.com/a/35974742/866930
def to_fwf(df, fname):
    content = tabulate(df.values.tolist(), list(df.columns), tablefmt="plain", floatfmt='.5f')
    open(fname, 'w').write(content)


pd.DataFrame.to_fwf = to_fwf


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


def load_manuscript_classifications():
    """This function loads the parameters and classifications posted in the DAmFRET manuscript.
    Namely, the parameters and classifications in Table S1 and Table S2.

    @return tuple: a 2-tuple comprised of pandas.DataFrame objects corresponding to the SI tables
                   of Table S1 (synthetic data), and Table S2 (real data) respectively.
    """
    synthetic = pkgutil.get_data('damfret_classifier', 'TableS1.csv')
    real = pkgutil.get_data('damfret_classifier', 'TableS2.csv')
    synthetic_data = pd.read_csv(io.BytesIO(synthetic))
    real_data = pd.read_csv(io.BytesIO(real))
    return synthetic_data, real_data


def generate_default_config():
    """A convenience function to generate a default config."""
    config_text = pkgutil.get_data('damfret_classifier', 'default_config.yaml')
    with open('config.yaml', 'w') as cfile:
        cfile.write(config_text.decode('utf-8'))


# ---------------------------------------------------------------------------------------------------------------------


def create_directory_if_not_exist(directory):
    """Create a directory if it does not exist and do not through an error if the `directory` already exists."""
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return os.path.isdir(path)



def shannon_entropy(histogram2d):
    """Calculate the Shannon Entropy of a 2D numpy array, `histogram2d`. This works by creating a probability distribution
    from the underlying 2D array and analyzing its Shannon Entropy.

    @param histogram2d (np.ndarray): A 2D array which will be analyzed.
    @return entropy (float): The Shannon Entropy calculated across the entire array.

    Ref: See this magnificent link: https://stackoverflow.com/questions/42683287/python-numpy-shannon-entropy-array
    """
    prob_distribution = histogram2d / histogram2d.sum()

    log_p = np.log(prob_distribution)
    log_p[log_p == -np.inf] = 0  # remove the infs
    entropy = -np.sum(prob_distribution * log_p)
    return entropy


def load_settings(yaml_filename):
    """This function loads a YAML configuration file into a dictionary object.

    @param yaml_filename: The filename of the YAML config file to load.
    @return dict: a dictionary populated as key / value pairs from the YAML config file.
    """
    with open(yaml_filename) as yfile:
        return yaml.safe_load(yfile)


def read_original_data(filename, low_conc_cutoff, high_conc_cutoff):
    """This is a convenience function will provides the user with the ability to load a given 
    dataset and apply the cutoffs (`low_conc_cutoff` and `high_conc_cutoff`) according to the 
    algorithm used.
    """
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


def remove_genes_and_replicates_below_count(genes_table, minimum_cell_count, construct_or_gene_column):
    """This is a helper function to prepare an input plasmid list into a table for easier use and
    management. This function is best paired with `validate_gene_replicates`. That fuction should be
    run after this.

    @param genes_table (pandas.DataFrame):  A table containing the genes, replicates, and well files.
    @param minimum_cell_count (int):        The minimum number of cell measurements that a data file
                                            must have in order to be kept.
    @param construct_or_gene_column (str):  The column to use for selection. Can be either `gene` or
                                            `construct`.
    @return tuple:                          A 2-tuple containing the truncated table, and a list 
                                            containing the genes / constructs that were excluded.

    """
    table = genes_table.copy()
    allowed = 'gene,construct'.split(',')
    if construct_or_gene_column not in allowed:
        raise RuntimeError('Column selection "{}" not allowed. Needs to be one of "{}".'.format(construct_or_gene_column, ', '.join(allowed)))
    search_col = construct_or_gene_column[:]

    df = table[table['counts'] <= minimum_cell_count]
    to_exclude = set(df[search_col].to_numpy().tolist())
    for item in to_exclude:
        match = table[table[search_col] == item]
        table.drop(match.index, inplace=True)
    table.reset_index(drop=True, inplace=True)  # Renumber the indices of the rows kept for simplicity.
    excluded = list(sorted(to_exclude))
    return table, excluded


def validate_gene_replicates(genes_table, num_replicates, drop_extraneous=True, drop_if_fewer=True):
    """This is a helper function which is meant to examine a given genes table derived / populated
    via `create_genes_table` and determine if there are genes which do not have the expected number
    of replicates. By default extraneous replicates are dropped. Genes with less than the expected
    number are dropped automatically by default.

    @param genes_table (pandas.DataFrame):  A table containing the genes, replicates, and well files.
    @param num_replicates (int):            The number of replicates to validate against.
    @param drop_extraneous (bool):          Whether or not to drop any replicates exceeding 
                                            `num_replicates` (default = True).
    @param drop_if_fewer (bool):            Whether or not to drop genes if their replicates are
                                            less than the expected number `num_replicates` (default=True).
    @return pandas.DataFrame:   The truncated `DataFrame`.
    """
    columns_order = 'construct,replicate,gene,well_file,plasmid,counts,AA_sequence'.split(',')
    table = genes_table.copy()
    
    genes = set(table['gene'].to_numpy().tolist())
    counts = OrderedDict()
    droppable = list()
    truncatable = list()
    for gene in sorted(genes):
        df = table[table['gene'] == gene]
        if len(df) != num_replicates:
            counts[gene] = len(df)

            if len(df) < num_replicates:
                droppable.append(df)
                if drop_if_fewer:
                    table.drop(df.index, inplace=True)
            else:
                sel = df.iloc[num_replicates:]
                if drop_extraneous:
                    truncatable.append(sel)
                    table.drop(sel.index, inplace=True)
    table.reset_index(drop=True, inplace=True)  # Renumber the indices of the rows kept for simplicity.

    dropped = pd.DataFrame()
    truncated = pd.DataFrame()

    if len(droppable) > 0:
        dropped = pd.concat(droppable)
        print('Some replicates were dropped. Those were saved to "dropped.tsv"...')
        dropped[columns_order].to_fwf('dropped.tsv')

    if len(truncatable) > 0:
        truncated = pd.concat(truncatable)
        print('Some replicates were truncated. Those were saved to "truncated.tsv"...')
        truncated[columns_order].to_fwf('truncated.tsv')
    
    message = list()
    header1 = '\nWarning: The following genes do not match the expected number ({}).'.format(num_replicates)
    message.append(header1)

    if drop_if_fewer:
        header2 = 'Genes with replicates < {r} were dropped.'.format(r=num_replicates)
        message.append(header2)
    
    if drop_extraneous:
        header3 = 'Genes with replicates > {r} were truncated to {r}:'.format(r=num_replicates)
        message.append(header3)

    for gene in counts:
        gene_counts = 'GENE: {}, ORIGINAL NUM REPLICATES: {}'.format(gene, counts[gene])
        message.append(gene_counts)

    if len(counts) > 0:
        error_message = '\n'.join(message)
        print(error_message)
    return table


def create_genes_table(config, plasmid_csv_filename, savename):
    """Given an input plasmid_csv filename and algorithm configuration, create a more comprehensive TSV dataset 
    for easier use. (TSVs are human and machine-readable.) This works by reading in the plasma well files and
    counting the number of cell measurements and adding that information to an extended table derived from the
    plasmid_csv. Other attributes are added such as the construct and replicate numbers. Those numbers should
    not be used for comparison as they are not fixed. They are mainly used for improving readability and 
    consulting.

    @param config (Config):             A `Config` instance which contains all the parameters for the project.
    @param plasmid_csv_filename (str):  A CSV file containing plasmid IDs, genes, sequences, and their well files.
    @param savename (str):              The name of the output TSV file.

    This function is meant to be run prior to the execution of the driver script which calls the function
    `classify_datasets`.
    """
    expected_columns = 'well_file,plasmid,gene,AA_sequence'.split(',')

    df = pd.read_csv(plasmid_csv_filename)
    actual_columns = list(df.keys())
    common = set(actual_columns).intersection(expected_columns)
    if len(common) != len(expected_columns):
        raise RuntimeError('Expected columns "{}" not found. Exiting.'.format(', '.join(expected_columns)))

    # reorder plasmid table by gene:
    all_genes = set(df['gene'].to_numpy().tolist())

    all_selections = list()
    construct = 1
    for gene in sorted(all_genes):
        sel = df[df['gene'] == gene].copy()
        replicates  = list()
        constructs  = list()
        counts      = list()
        replicate   = 1
        for _index, row in sel.iterrows():
            well_file  = row['well_file']

            # first check that the well file exists with a leading zero.
            data_filename = os.path.join(config.data_dir, config.filename_format.format(well_name=well_file))
            if not os.path.exists(data_filename):
                well            = int(well_file[1:])  # works for `A09` and `A9`
                well_name       = '%s%d' % (well_file[0], well)  # reformat to match CSV data files name format.
                data_filename   = os.path.join(config.data_dir, config.filename_format.format(well_name=well_name))
            
            data        = pd.read_csv(data_filename)
            cell_counts = len(data)

            replicates.append(replicate)
            constructs.append(construct)
            counts.append(cell_counts)
            replicate += 1
        
        sel['replicate']    = replicates
        sel['construct']    = constructs
        sel['counts']       = counts

        all_selections.append(sel)
        construct += 1

    columns_order = 'construct,replicate,gene,well_file,plasmid,counts,AA_sequence'.split(',')
    reordered_df = pd.concat(all_selections)
    reordered_df[columns_order].to_fwf(savename)
    return reordered_df


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