import numpy as np
from pathlib import Path
from datetime import datetime
from damfret_classifier.utils import load_settings


class Config(object):
    """This object is designed primarily for convenience as it stores, validates, and
    auto-calculates certain variables such the number of bins for subsequent use by the
    other modules.
    """
    def __init__(self, settings_filename=None, config_identifier=None, **kwargs):
        # Parameters for the algorithm
        self.low_conc_cutoff    = kwargs.get('low_conc_cutoff',                 0.0)
        self.high_conc_cutoff   = kwargs.get('high_conc_cutoff',                0.0)
        self.low_conc           = kwargs.get('low_conc',                        0.0)
        self.high_conc          = kwargs.get('high_conc',                       0.0)
        self.low_fret           = kwargs.get('low_fret',                        0.0)
        self.high_fret          = kwargs.get('high_fret',                       0.0)
        self.conc_bin_width     = kwargs.get('conc_bin_width',                  0.0)
        self.fret_bin_width     = kwargs.get('fret_bin_width',                  0.0)
        self.num_replicates     = kwargs.get('num_replicates',                  0)
        self.number_of_bins_xy  = kwargs.get('number_of_bins_xy',               0)
        self.drop_cell_counts   = kwargs.get('drop_cell_counts',                0)

        # Directories
        self.work_dir           = kwargs.get('work_directory',                  None)
        self.plots_dir          = kwargs.get('plots_directory',                 None)
        self.data_dir           = kwargs.get('data_directory',                  None)
        self.filename_format    = kwargs.get('filename_format',                 None)

        # Plotting
        self.plot_gaussian      = kwargs.get('generate_gaussian_fits_plots',    False)
        self.plot_logistic      = kwargs.get('generate_logistic_fits_plots',    False)
        self.plot_fine_grids    = kwargs.get('generate_fine_grids_plots',       False)
        self.plot_rsquared      = kwargs.get('generate_rsquared_fits_plots',    False)
        self.fine_grid_xlim     = kwargs.get('fine_grid_xlim',                  None)
        self.fine_grid_ylim     = kwargs.get('fine_grid_ylim',                  None)
        self.plot_type          = kwargs.get('plot_type',                       'png')

        # Config identifier data
        self.settings_filename  = settings_filename
        self.config_identifier  = config_identifier
        
        # Populated when `self._populate_bins_variables()` is called.
        self.num_conc_bins  = 0
        self.num_fret_bins  = 0
        self.conc_bins      = None
        self.fret_bins      = None
        self.fg_conc_edges  = None
        self.fg_fret_edges  = None

        if settings_filename is not None and config_identifier is not None:
            self.load_config_from_file(settings_filename, config_identifier)
        else:
            self._populate_bins_variables()
            self.validate()


    def _check_if_null(self, parameter):
        value = getattr(self, parameter)
        if value is None:
            message = """Parameter "{}" has not been initialized.
            Check the values of `num_conc_bins`, `num_fret_bins`, `high_conc`, `low_conc`, `conc_bin_width` or `fret_bin_width`.""".format(parameter)
            raise RuntimeError(message)


    def _check_if_zero(self, parameter):
        value = getattr(self, parameter)
        if type(value) is int:
            if value == 0:
                raise RuntimeError('Parameter "{}" needs to be larger than `0`.'.format(parameter))
        elif type(value) is float:
            if value == 0.0:
                raise RuntimeError('Parameter "{}" needs to be larger than `0.0`.'.format(parameter))


    def _check_limits_var(self, limit_variable):
        value = getattr(self, limit_variable)
        if value is not None:
            if len(value) != 2:
                raise RuntimeError('An upper and lower limit ONLY is required for `{}`.'.format(limit_variable))


    def _now_timestamp(self):
        now = datetime.now()
        s = now.strftime('%Y-%m-%d_%H-%M-%S')
        return s


    def _create_directory_if_not_exists(self, directory, default_name, timestamp_string):
        if directory is not None:
            path = Path(directory)
            if not path.exists():
                print('Warning: {} path "{}" not found. This path will be created.'.format(default_name, path))
                path.mkdir(parents=True, exist_ok=False)
                print('User-specified {} directory created: "{}".'.format(default_name, path))

            if not path.is_dir():
                raise RuntimeError('The supplied path: "{}" is not a directory. Exiting.'.format(path))
        else:
            name = '{}---{}'.format(default_name, timestamp_string)
            path = Path(Path.cwd(), name)
            try:
                path.mkdir(parents=True, exist_ok=False)
                print('Default {} directory created: "{}".'.format(default_name, path))
            except FileExistsError:
                raise RuntimeError('Could not create default {} path: "{}". Directory already exists.'.format(default_name, path))
        return path


    def _validate_limits(self):
        if self.low_conc_cutoff < self.low_conc:
            raise RuntimeError('The lower concentration cutoff cannot be less than the low concentration limit.')

        if self.high_conc_cutoff < self.low_conc:
            raise RuntimeError('The lower concentration cutoff cannot be less than the low concentration limit.')

        conc_diff = self.high_conc - self.low_conc
        fret_diff = self.high_fret - self.low_fret
        if conc_diff <= 0:
            raise RuntimeError('The `low_conc` cannot be >= `high_conc`.')

        if fret_diff <= 0:
            raise RuntimeError('The `low_fret` cannot be >= `high_fret`.')


    def _validate_plot_type(self):
        if self.plot_type is None:
            self.plot_type = 'png'
        else:
            # Tiff can be supported, but requires the installation of Pillow, the replacement for the obsolete PIL package.
            allowed = 'png,pdf,svg,eps'.split(',')
            if self.plot_type.lower() not in allowed:
                raise RuntimeError('Unsupported plot extension: "{}". Try one of: "{}".'.format(self.plot_type, ', '.join(allowed)))


    def _validate_directories(self):
        # Check for work and plot path existence. Create if not found.
        now_str = self._now_timestamp()
        self.work_dir   = self._create_directory_if_not_exists(self.work_dir, 'work', now_str)
        self.plots_dir  = self._create_directory_if_not_exists(self.plots_dir, 'plots', now_str)

        if self.data_dir is None:
            raise RuntimeError('No data directory `data_dir` has been supplied in the settings. Exiting.')


    def _validate_filename_format(self):
        if type(self.filename_format) is not str:
            raise RuntimeError('The `filename_format` must be a string. Exiting.')
        else:
            if '{well_name}' not in self.filename_format:
                raise RuntimeError('The `filename_format` must contain the template variable: "{well_name}".')


    def validate(self):
        # Check the conc and fret limits.
        self._validate_limits()
        
        # Check and validate the values / limits of the important variables.
        self._check_if_zero('drop_cell_counts')
        self._check_if_zero('num_conc_bins')
        self._check_if_zero('num_fret_bins')
        self._check_if_zero('number_of_bins_xy')

        self._check_if_zero('high_conc')
        self._check_if_zero('num_replicates')
        self._check_if_zero('conc_bin_width')
        self._check_if_zero('fret_bin_width')

        self._check_if_null('conc_bins')
        self._check_if_null('fret_bins')
        self._check_if_null('fg_conc_edges')
        self._check_if_null('fg_fret_edges')

        self._check_limits_var('fine_grid_xlim')
        self._check_limits_var('fine_grid_ylim')

        # Check that the filename_format is not Null.
        self._validate_filename_format()

        # Check for data, work, and plot directories existence. Create work and plot directories if not found.
        self._validate_directories()
        
        # Check for valid plot types.
        self._validate_plot_type()

    
    def _populate_bins_variables(self):
        self.num_conc_bins = int((self.high_conc - self.low_conc)/self.conc_bin_width) + 1
        self.num_fret_bins = int((self.high_fret - self.low_fret)/self.fret_bin_width) + 1

        self.conc_bins = np.linspace(self.low_conc, self.high_conc, self.num_conc_bins)
        self.fret_bins = np.linspace(self.low_fret, self.high_fret, self.num_fret_bins)
        self.fg_conc_edges = np.linspace(self.low_conc, self.high_conc, self.number_of_bins_xy + 1)
        self.fg_fret_edges = np.linspace(self.low_fret, self.high_fret, self.number_of_bins_xy + 1)


    def load_config_from_file(self, settings_filename, config_identifier):
        settings = load_settings(settings_filename)
        c = settings[config_identifier]

        self.low_conc_cutoff    = c['low_conc_cutoff']
        self.high_conc_cutoff   = c['high_conc_cutoff']
        self.low_conc           = c['low_conc']
        self.high_conc          = c['high_conc']
        self.low_fret           = c['low_fret']
        self.high_fret          = c['high_fret']
        self.conc_bin_width     = c['conc_bin_width']
        self.fret_bin_width     = c['fret_bin_width']
        self.num_replicates     = c['num_replicates']
        self.number_of_bins_xy  = c['number_of_bins']
        self.work_dir           = c['work_directory']
        self.plots_dir          = c['plots_directory']
        self.data_dir           = c['data_directory']
        self.filename_format    = c['filename_format']
        self.drop_cell_counts   = c['drop_cell_counts']
        self.plot_gaussian      = c['generate_gaussian_fits_plots']
        self.plot_logistic      = c['generate_logistic_fits_plots']
        self.plot_fine_grids    = c['generate_fine_grids_plots']
        self.plot_rsquared      = c['generate_rsquared_fits_plots']
        self.plot_type          = c['plot_type']
        self.fine_grid_xlim     = c['fine_grid_xlim']
        self.fine_grid_ylim     = c['fine_grid_ylim']
        
        self._populate_bins_variables()
        self.validate()


    def __repr__(self):
        params = 'settings_filename,config_identifier'.split(',')
        params += 'low_conc,high_conc,low_fret,high_fret,low_conc_cutoff,high_conc_cutoff,conc_bin_width'.split(',')
        params += 'fret_bin_width,num_replicates,number_of_bins_xy,plots_dir,data_dir,drop_cell_counts'.split(',')
        params += 'filename_format,plot_gaussian,plot_logistic,plot_fine_grids,plot_rsquared,plot_type'.split(',')
        params += 'fine_grid_xlim,fine_grid_ylim'.split(',')

        params_and_values = list()
        for param in params:
            value = getattr(self, param)
            if type(value) is str:
                param_string = '{}="{}"'.format(param, value)
            else:
                param_string = '{}={}'.format(param, value)
            params_and_values.append(param_string)
        
        params_string = ', '.join(params_and_values)
        return 'Config({})'.format(params_string)


    def __str__(self):
        return str(repr(self))
