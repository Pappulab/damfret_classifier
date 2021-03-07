import os
import re
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
           'create_genes_table', 'start_process', 'initialize_pool', 'parallelize', 'generate_default_config', 'to_fwf',
           'clamp_data', 'find_subdirectories', 'read_manioc_config', 'validate_manioc_directory_tree',
           'check_if_data_within_limits', 'parse_manioc_timestamp', 'determine_if_manioc_project', 'determine_well_name',
           'generate_wells_filename_from_directory_contents', 'find_subdirectories_with_files_for_analysis',
           'export_raw_synthetic_data']


# ---------------------------------------------------------------------------------------------------------------------
# Pandas utils


# Supremely useful answer to convert a DataFrame to a TSV file: https://stackoverflow.com/a/35974742/866930
def to_fwf(df, fname):
    content = tabulate(df.values.tolist(), list(df.columns), tablefmt="plain", floatfmt='.5f')
    open(fname, 'w').write(content)


pd.DataFrame.to_fwf = to_fwf


# ---------------------------------------------------------------------------------------------------------------------
# Loader and generator utils


def load_raw_synthetic_data():
    """This function loads the raw synthetic data referenced in the DAmFRET manuscript as a
    `pandas.DataFrame()` object.

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


def export_raw_synthetic_data(savepath, extension):
    """This function loads and exports the raw synthetic data used within the DAmFRET manuscript
    into individual files / "wells", the format of which depends on the extension requested.
    
    @param savepath (str):  The location where to save the exported synthetic data.
    @param extension (str): The extension to use when exporting the filenames of the synthetic data.
    @return None
    """
    allowed = 'txt,dat,tsv,csv,ascii,xlsx'.split(',')
    allowed_for_tsv = 'txt,dat,ascii'.split(',')
    if extension not in allowed:
        raise RuntimeError('Extension not supported for exporting. Please use one of: {}.'.format(allowed))

    synthetic_data = load_raw_synthetic_data()
    total_datasets = len(synthetic_data)
    absolute_savepath = Path(savepath).absolute()
    for index, dataset in enumerate(synthetic_data, start=1):
        data_savepath = os.path.join(absolute_savepath, '{dataset}.{ext}'.format(dataset=dataset, ext=extension))
        df, _original_name = synthetic_data[dataset]

        if extension in allowed_for_tsv:
            to_fwf(df, data_savepath)
        elif extension == 'csv':
            df.to_csv(data_savepath)
        elif extension == 'xls' or extension == 'xlsx':
            df.to_excel(data_savepath)
        else:
            raise RuntimeError('Extension not supported.')
        print('Export of synthetic dataset ({:0>3d} / {}): "{}" complete.'.format(index, total_datasets, data_savepath))


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
# Filesystem utils


def create_directory_if_not_exist(directory):
    """Create a directory if it does not exist and do not through an error if the `directory` already exists.

    @param directory (str): The path of the directory which will be created.
    @return bool: Whether or not the path created is a directory.
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return os.path.isdir(path)


def find_subdirectories(pathname):
    """Identify the subdirectories within a provided path, and return their full path names.

    @pathname (str): The location of the directory that will be searched.
    @return subdirectories (list): A list of all the subdirectories located within the provided path.
    """
    # Expand the pathname in case it contains `~/` i.e. a relative path to the user's home directory.
    expanded_path = os.path.expanduser(pathname)
    path = Path(expanded_path)
    if not path.is_dir():
        raise RuntimeError('The path supplied "{}" is a file - please pass a path name instead.'.format(pathname))

    subdirectories = [p for p in path.glob('**/') if p.is_dir()]
    return subdirectories


def determine_well_name(raw_well_name, zero_pad=False):
    """Use a regex to extract the name of the well from that provided from the plasmid table
    or other equivalent resource. This is the most robust method and more future proof overall."""
    well_regex = re.compile('([A-Z]+)(\\d+)')
    match = well_regex.search(raw_well_name)
    if match is None:
        raise RuntimeError('No well entry found for well: {}'.format(raw_well_name))

    groups = match.groups()
    base_well_name = groups[0]
    well_num = groups[1]

    # reformat to match the CSV name format (i.e. not zero-padded) - hence the use of `int`.
    well_name = '{base_well}{well_num}'.format(base_well=base_well_name, well_num=int(well_num))
    if zero_pad:
        well_name = '{}{:0>2}'.format(base_well_name, int(well_num))
    return well_name



def find_subdirectories_with_files_for_analysis(pathname, filename_format):
    """This function examines a given path `pathname` to determine if there
    are subdirectories which contain data for analysis. Each sub-directory 
    will be configured as an individual session and analyzed accordingly.

    Relative paths can be used - e.g. `'.'`. Note that in such an instance
    if there are data files with the same extension as contained in
    `filename_format`, and even if the files have the typical wells filename
    naming scheme, those files will not be included. Only sub-directories
    from the path provided will be enumerated.

    This function should be run before 
    `generate_wells_filename_from_directory_contents`, and can be considered
    as its companion function.

    @param pathname (str):          The location of the directory to analyze.
                                    Relative paths such `'.'` are also
                                    supported.

    @param filename_format (str):   The `filename_format` of the individual
                                    wells. Typically this parameter should
                                    be populated from `filename_format`
                                    from the YAML config.

    @return OrderedDict:            An `OrderedDict` in which the keys
                                    are the absolute path names of the
                                    sub-directories containing files that
                                    match the `filename_format`. The
                                    corresponding values are lists 
                                    containing the absolute paths of 
                                    files within that sub-directory.
    """
    dirs = find_subdirectories(pathname)
    _base_filename, filetype = os.path.splitext(filename_format)

    # Screen the names of the sub-directories found to ensure that we
    # don't examine those from previous analysis runs as indicated
    # by the presence of `work` and `logs`. We exclude any directory
    # which contains `raw_data` within the immediate sub-directory
    # as those files have not yet been gated.
    cwd = str(Path().absolute())
    paths_of_interest = list()
    for directory in dirs:
        abs_path = str(directory.absolute())
        if abs_path == cwd:
            continue
        if 'work---' in abs_path:
            continue
        if 'logs---' in abs_path:
            continue
        if str(directory).startswith('raw_data'):
            continue
        paths_of_interest.append(abs_path)

    # Next, we examine each sub-directory of interest and identify
    # which files have the same extension as the input `filename_format`
    # (usually matches that of `filename_format` from the YAML config).
    #
    # If these subdirectories contain files with the same extension
    # as the `filename_format` they are kept, but will ultimately be
    # excluded in the companion function to this one 
    # (`generate_wells_filename_from_directory_contents`) as they will
    # not match the known naming format for wells (see: 
    # `determine_well_name`).
    dir_contents = OrderedDict()
    for directory in paths_of_interest:
        dir_contents[directory] = list()
        for fname in os.listdir(directory):
            path = str(Path(directory).joinpath(fname).absolute())
            if path.endswith(filetype) and 'work---' not in path and 'logs---' not in path:
                dir_contents[directory].append(path)

    # Finally, we whittle the contents to only include the locations
    # that are non-empty (according to the criteria above).
    dirs_of_interest = OrderedDict()
    for directory in sorted(dir_contents):
        contents = dir_contents[directory]
        if len(contents) > 0:
            dirs_of_interest[directory] = contents

    return dirs_of_interest


def generate_wells_filename_from_directory_contents(directory_contents, savename):
    """This function is used to examine the contents collected from the post-search and
    analysis of a provided directory tree (i.e. `project_directory` in the config). Then,
    it saves and returns those contents in a more convenient format that can be used for
    additional processing or book-keeping.

    By default, the directory contents are formatted as a `pandas.DataFrame` object and
    saved to the provided `savename`.

    This function use is meant to accompany `find_subdirectories_with_files_for_analysis`.

    @param directory_contents (list):   This list contains the processed contents derived
                                        from `find_subdirectories_with_files_for_analysis`.

    @param savename (str):              The savepath to write the reformatted contents 
                                        (`pandas.DataFrame`) out to.

    @return pandas.DataFrame:           A `pandas.DataFrame` object which contains the
                                        directory contents (usually well files) in a
                                        convenient, zero-padded, and sorted manner. The
                                        only column in the DataFrame is `well_file`.

    The purpose of this design is such that in cases where multiple runs are stored under
    an input `project_directory` root, there may be different wells. By having identifying 
    which sub-directories have wells files for analysis via 
    `find_subdirectories_with_files_for_analysis`, those locations can then be processed
    with this function. Hence, individual `wells_filename` can be determined and the
    analysis can proceed as expected.
    """
    # This expects that the wells_filename is empty.
    data = OrderedDict()
    data['well_file'] = list()
    for filename in sorted(directory_contents):
        fname = os.path.basename(filename)
        base_name, _extension = os.path.splitext(fname)
        try:
            padded_name = determine_well_name(base_name, zero_pad=True)
        except RuntimeError as e:
            print('Excluding file which does not match the known format of well filenames: "{}".'.format(filename))
            print(e)
            continue
        data['well_file'].append(padded_name)

    dfo = OrderedDict(sorted(data.items()))
    df = pd.DataFrame(dfo)
    df.to_csv(savename, index=False)
    return df


# ---------------------------------------------------------------------------------------------------------------------
# MANIOC utils


def read_manioc_config(config_filename):
    """This function reads in a MANIOC config file, and parses the contents into an `OrderedDict`
    for easier use and access. This object can then be used for validating the existence of the
    paths.

    @param config_filename (str): The name of the config file which will be parsed into an
                                  `OrderedDict` object.
    @return config (OrderedDict): An `OrderedDict` in which the key / value pairs have
                                  been extracted from the MANIOC config file.
    """
    config = OrderedDict()

    # Check that the config file exists; exit otherwise.
    if not os.path.exists(config_filename):
        raise RuntimeError('MANIOC config file "{}" not found. Exiting.'.format(config_filename))

    # Read the data in as a bunch of raw lines.
    with open(config_filename, 'r') as config_file:
        data = config_file.readlines()

    # In the first pass, remove the comments.
    cleaned_data = list()
    for line in data:
        if not line.startswith('#') or '#' not in line:
            cleaned_data.append(line.strip())
        else:
            # Remove the comment data (i.e. all characters after '#')
            # in case a line contains an appended in-line comment.
            index   = line.index('#')
            ln      = line[:index].strip()
            if len(ln) > 0:
                cleaned_data.append(ln)

    # In the second pass, populate the config based on the cleaned data.
    for line in cleaned_data:
        setting, option = line.split(':=')
        config[setting] = option

    # Our faux config object is now populated and available for use.
    return config


def parse_manioc_timestamp(time_str):
    """Read an input string from MANIOC and extract the datetime from it. Useful for populating log files
    and other related house-keeping tasks.

    @param time_str (str): The string that may contain a datetime string (in an ISO-like format). If
                           found, it will be extracted and converted to a datetime object which will
                           be returned.

    @return tuple:         A 2-tuple containing a boolean and a datetime object if a datetime string
                           is identified. Otherwise, the 2-tuple returns a boolean (False) and None.
    """
    parsed          = False
    datetime_obj    = None

    # This matches against MANIOC's ISO-like datetime format i.e. `YYYY-mm-DD-HH-MM-SS`.
    regex = re.compile('(\\d{4})-(\\d{2})-(\\d{2})-(\\d{2})-(\\d{2})-(\\d{2})')
    match = regex.search(time_str)  # See: https://docs.python.org/3/library/re.html#search-vs-match

    # Convert the regex match (tuple), if found, to a string, and parse it to a datetime object.
    if match is not None:
        date_tuple      = match.groups()
        datetime_obj    = datetime.strptime('-'.join(date_tuple), '%Y-%m-%d-%H-%M-%S')
        parsed          = True

    # Finally, return it.
    return parsed, datetime_obj


def validate_manioc_directory_tree(pathname):
    """This function checks and verifies the existences of the `manioc.results` sub-directories as read from the
    input `manioc.config` filename. If the required sub-directories are not found, the program raises an error
    and exits.

    @param pathname (str):              The directory whose contents will be examined and checked.
    @return paths_of_interest (list):   A list containing paths that are likely to contain data for subsequent
                                        analysis.

    If this succeeds, subsequent analysis can proceed.
    """
    # Read in the MANIOC config and consult its variables for use.
    root_path           = Path(pathname).expanduser().absolute()
    manioc_path         = root_path.joinpath('manioc.config')
    manioc_config       = read_manioc_config(manioc_path)
    manioc_results_root = root_path.joinpath(manioc_config['resultsDir'], manioc_config['rawDir'])

    # Populate the base directories which will be searched.
    # Common directories that could be found are `raw_data`, `analysis`, and `manioc.results`.
    #
    # Only `raw_data` and `manioc.results` are required. However, as `raw_data` is actually
    # populated from the variable `rawDir` in the MANIOC config, we consult that directly
    # since the value could change; `manioc.results` appears to be fixed.
    raw_dir                 = str(Path(manioc_config['rawDir']))
    required_directories    = '{raw_dir},manioc.results'.format(raw_dir=raw_dir)
    subdirectories          = find_subdirectories(root_path)
    required_subdirectories = {p:root_path.joinpath(p) for p in required_directories.split(',')}

    # Check that the required base subdirectories are found in the search path.
    for dirname in required_subdirectories:
        directory = required_subdirectories[dirname]
        if directory not in subdirectories:
            raise RuntimeError('Required base directory "{}" not found in root path: "{}"'.format(dirname, root_path))

    # Now check whether the tree from `raw_data` is populated in `manioc.results`
    # since if this is validated, it means that we can proceed with the analysis.
    # Recall that MANIOC creates an exact duplicate of the directory tree under
    # `raw_data` but prepended with the `resultsDir` path from the MANIOC config.
    raw_data_dir    = required_subdirectories[raw_dir]
    relative_paths  = list()
    for subdir in subdirectories:
        str_raw_data_dir    = str(raw_data_dir)
        str_subdir          = str(subdir)

        # Check for instances where the relative path is contained in the
        # name of the paths being searched. We exclude the exact match as
        # it is superfluous; hence the use of `!=`.
        if str_raw_data_dir != str_subdir and str_raw_data_dir in str_subdir:
            relative_path = os.path.relpath(subdir, raw_data_dir)
            relative_paths.append(relative_path)

    paths_of_interest = list()
    for relative_path in relative_paths:
        manioc_results_path = manioc_results_root.joinpath(relative_path).resolve()
        if not manioc_results_path.exists():
            raise RuntimeError('Required MANIOC results directory not found: "{}".'.format(manioc_results_path))

        # Extract only the subdirectories containing timestamps. These are the subdirectories
        # which likely contain processed data ready for analysis.
        parsed, _date = parse_manioc_timestamp(relative_path)  # `_date` is ignored - hence the leading underscore.
        if parsed:
            paths_of_interest.append(manioc_results_path)
    return paths_of_interest


def determine_if_manioc_project(project_directory):
    """This function determines if a supplied path to a `project_directory` contains MANIOC output. This is done
    by checking for the existence of `'manioc.results'` and `'raw_data'` sub-directories. If these are present
    the directory is a MANIOC project.

    @param project_directory (str): The name of the project directory to check.
    @return tuple:                  A 2-tuple corresponding to whether or not the directory has a MANIOC layout,
                                    and, a list of the immediate sub-directories located in the provided project
                                    directory.
    """
    project_dir = Path(project_directory).expanduser().resolve()
    contains_manioc_results = False
    contains_raw_data = False
    immediate_sub_directories = list()
    for name in os.listdir(project_dir):
        path = os.path.join(project_dir, name)
        if os.path.isdir(path):
            immediate_sub_directories.append(path)
            if 'manioc.results' in name:
                contains_manioc_results = True
            elif 'raw_data' in name:
                contains_raw_data = True
    valid = contains_manioc_results and contains_raw_data
    return valid, immediate_sub_directories


# ---------------------------------------------------------------------------------------------------------------------
# Miscellaneous utils


def clamp_data(data, low_conc_cutoff, high_conc_cutoff):
    """This function restricts the input `data` to be contained within the limits:
    `low_conc_cutoff` and `high_conc_cutoff`.

    @param data (pandas.DataFrame):     The raw input data yet to be histogrammed.
    @param low_conc_cutoff (float):     The lower cutoff limit. Values below this will be dropped.
    @param high_conc_cutoff (float):    The upper cutoff limit. Values above this will be dropped.

    @return pandas.DataFrame:           The updated DataFrame with the values outside of the limits
                                        removed.
    """
    data = data.dropna()  # remove rows containing NaN values
    data = data[data['concentration'] >= low_conc_cutoff]
    data = data[data['concentration'] <= high_conc_cutoff]
    return data


def check_if_data_within_limits(data, low_conc_cutoff, high_conc_cutoff):
    """This function checks whether valid data exists within the limits provided. It uses the min
    and max of the raw concentration data (i.e. pre-clamped) as an back of the envelope check for
    validity. This is necessary as data outside of these limits will be dropped, and the
    classification may not proceed, or in some cases be inaccurate. It should be noted that minimum
    concentration will be negative, but that's alright as the system is very noisy at low concentration.

    @param data (pandas.DataFrame):                 The raw input data yet to be histogrammed.
    @param low_conc_cutoff (float):                 The lower cutoff limit. Values below this
                                                    will be dropped.
    @param high_conc_cutoff (float):                The upper cutoff limit. Values above this
                                                    will be dropped.

    @return valid (bool), min (float), max (float): Values corresponding to the check which could
                                                    be used to isolate the current data set for
                                                    additional processing.
    """
    data = data.dropna()
    min_conc = min(data['concentration'])
    max_conc = max(data['concentration'])

    valid = False
    if max_conc <= high_conc_cutoff and max_conc >= low_conc_cutoff:
        valid = True

    return valid, min_conc, max_conc


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


def read_original_data(filename, low_conc_cutoff, high_conc_cutoff, apply_cutoff=True):
    """This is a convenience function will provides the user with the ability to load a given
    dataset and apply the cutoffs (`low_conc_cutoff` and `high_conc_cutoff`) according to the
    algorithm used.
    """
    data = pd.read_csv(filename, usecols=['Acceptor-A', 'FRET-A'])
    counts = len(data)  # Record the original number of points

    df = pd.DataFrame()
    df['concentration'] = np.log10(data['Acceptor-A'])
    df['damfret'] = data['FRET-A']/data['Acceptor-A']

    # Remove extraneous data not found between the limits if requested.
    if apply_cutoff:
        df = df[df['concentration'] >= low_conc_cutoff]
        df = df[df['concentration'] <= high_conc_cutoff]
    df = df.dropna()  # Remove any rows with NaN values in the FRET columns.
    df.reset_index(drop=True, inplace=True)  # Renumber the indices of the rows kept for simplicity.

    return df, counts


def apply_cutoff_to_dataframe(data_df, low_conc_cutoff, high_conc_cutoff):
    """This is another convenience function to apply a cutoff on an already loaded DataFrame."""
    df = data_df.copy()
    counts = len(data_df)  # Record the original number of points

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
# Parallelization utils


def start_process(session_name):
    """This is just a notifier function to indicate that a worker process has been launched.
    Since this is meant for a pool, the pool reuses these processes."""
    session_log = logging.getLogger(session_name)
    session_log.info('Starting :: {}'.format(mp.current_process().name))


def initialize_pool(num_processes, initializer_function, session_name):
    """Initialize a multiprocessing pool based on the number of requested processes and
    some initializer function which is used primarily for notification."""
    num_procs = num_processes
    if num_processes is None:
        num_procs = mp.cpu_count()

    pool = mp.Pool(processes=num_procs, initializer=initializer_function, initargs=(session_name,))
    return pool


def parallelize(func, list_of_func_args, num_processes=None, session_name=None):
    """This function parallelizes calling a function with different argument values.

    @param func (func):                 The function reference which will be called.
    @param num_repeats (int):           The number of times to call the function.
    @param func_args (tuple | list):    The arguments to pass to the function call.
    @param num_processes (int):         The number of processes to use (default = all).
    @param session_name (str):          The session name / log that will be written to.

    @returns results (OrderedDict):     A dictionary of function arguments as keys, and
    a values of 2-tuples comprised of datetime instances and the function result. Note:
    the result is a lazy reference to an `ApplyAsync` result. To extract the values,
    they can be obtained by calling `.get()` on the result.

    Notes: calling the `ApplyAsync` object's get (`result.get()`) here will block, which
    defeats the entire purpose behind this configuration.
    """
    pool = initialize_pool(num_processes, start_process, session_name)

    results = OrderedDict()
    for func_args in list_of_func_args:
        async_result = pool.apply_async(func, args=func_args)
        start_time = datetime.now()
        func_result = (async_result, start_time)
        results[func_args] = func_result

    pool.close()
    pool.join()

    # To obtain the results as they arrive, we need to use `AsyncResult.get()` - i.e.
    # `result.get()`. However, doing this in the processing loop will block and never
    # yield performance increases above those using 1 CPU. So, to parallelize the
    # results call, we do the `get()` after the pool has been closed and joined.
    actual_results = OrderedDict()
    for fargs in results:
        async_result, start_time = results[fargs]
        result = async_result.get()
        current_time = datetime.now()
        actual_results[fargs] = (result, start_time, current_time)

    return actual_results
