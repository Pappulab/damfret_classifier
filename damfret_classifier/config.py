import numpy as np
import logging
import multiprocessing as mp
from pathlib import Path
from datetime import datetime
from damfret_classifier.utils import load_settings


class Config(object):
    """This object is designed primarily for convenience as it stores, validates, and
    auto-calculates certain variables such the number of bins for subsequent use by the
    other modules.
    """
    def __init__(self, settings_filename=None, **kwargs):
        """The main entry point and initializer. This is configured such that one can
        directly create the object from another config object by passing and populating
        the relevant parameter via the `kwargs` dictionary. This design is most useful
        when debugging."""

        # Book-keeping data
        self.settings_filename  = settings_filename
        self.session_name       = kwargs.get('session_name',                    'session')

        # Job control parameters
        self.random_seed        = kwargs.get('random_seed',                     None)
        self.num_processes      = kwargs.get('num_processes',                   1)
        self.nice_level         = kwargs.get('nice_level',                      0)

        # Parameters for the algorithm
        self.low_conc_cutoff    = kwargs.get('low_conc_cutoff',                 0.0)
        self.high_conc_cutoff   = kwargs.get('high_conc_cutoff',                0.0)
        self.low_conc           = kwargs.get('low_conc',                        0.0)
        self.high_conc          = kwargs.get('high_conc',                       0.0)
        self.low_fret           = kwargs.get('low_fret',                        0.0)
        self.high_fret          = kwargs.get('high_fret',                       0.0)
        self.conc_bin_width     = kwargs.get('conc_bin_width',                  0.0)
        self.fret_bin_width     = kwargs.get('fret_bin_width',                  0.0)
        self.number_of_bins_xy  = kwargs.get('number_of_bins_xy',               0)
        self.min_measurements   = kwargs.get('minimum_required_measurements',   0)

        # Important files and directories
        self.project_dir        = kwargs.get('project_directory',               None)
        self.work_dir           = kwargs.get('work_directory',                  None)
        self.logs_dir           = kwargs.get('logs_directory',                  None)
        self.filename_format    = kwargs.get('filename_format',                 None)
        self.wells_filename     = kwargs.get('wells_filename',                  None)
        self.move_to_pdir       = kwargs.get('move_to_project_directory',       None)

        # Plotting
        self.plot_gaussian      = kwargs.get('plot_gaussian_fits',              False)
        self.plot_logistic      = kwargs.get('plot_logistic_fits',              False)
        self.plot_fine_grids    = kwargs.get('plot_2d_histograms',              False)
        self.plot_rsquared      = kwargs.get('plot_rsquared_fits',              False)
        self.plot_skipped       = kwargs.get('plot_skipped_2d_histograms',      False)
        self.fine_grid_xlim     = kwargs.get('xlim_2d_histograms',              None)
        self.fine_grid_ylim     = kwargs.get('ylim_2d_histograms',              None)
        self.plot_type          = kwargs.get('plot_type',                       'png')

        # Populated when `self._populate_bins_variables()` is called.
        self.num_conc_bins  = 0
        self.num_fret_bins  = 0
        self.conc_bins      = None
        self.fret_bins      = None
        self.fg_conc_edges  = None
        self.fg_fret_edges  = None

        if settings_filename is not None:
            self.load_config_from_file(settings_filename)
        else:
            self._populate_bins_variables()
            self.validate()


    def _check_if_null(self, parameter):
        """Used during validation as a number of parameters, mainly those which are auto-generated
        using the provided limits (i.e. bins), should not be `None` (the default)."""
        value = getattr(self, parameter)
        if value is None:
            message = """Parameter "{}" has not been initialized.
            Check the values of `num_conc_bins`, `num_fret_bins`, `high_conc`, `low_conc`, `conc_bin_width` or `fret_bin_width`.""".format(parameter)
            raise RuntimeError(message)


    def _check_if_zero(self, parameter):
        """Used during validation as there are a few parameters which should be non-zero."""
        value = getattr(self, parameter)
        if type(value) is int:
            if value == 0:
                raise RuntimeError('Parameter "{}" needs to be larger than `0`.'.format(parameter))
        elif type(value) is float:
            if value == 0.0:
                raise RuntimeError('Parameter "{}" needs to be larger than `0.0`.'.format(parameter))


    def _check_limits_var(self, limit_variable):
        """This is helper function for validating the number of parameters passed as limits from the config."""
        value = getattr(self, limit_variable)
        if value is not None:
            if len(value) != 2:
                raise RuntimeError('An upper and lower limit ONLY is required for `{}`.'.format(limit_variable))


    def _now_timestamp(self):
        """Generate an ISO-like timestamp."""
        now = datetime.now()
        s = now.strftime('%Y-%m-%d_%H-%M-%S')
        return s


    def _create_directory_if_not_exists(self, directory, default_name, timestamp_string, save_in_project_dir=True):
        """Create a requested directory if it didn't exist prior. Raise an error if the directory exists since
        we want to avoid race conditions and cases where previous results may be overwritten."""
        session_log = logging.getLogger(self.session_name)
        if directory is not None:
            path = Path(directory).expanduser().absolute()
            if not path.exists():
                message1 = 'Warning: {} path "{}" not found. This path will be created.'.format(default_name, path)
                message2 = 'User-specified {} directory created: "{}".'.format(default_name, path)

                # `exist_ok = False` here mainly for preservation and minimizing issues
                # with race conditions
                path.mkdir(parents=True, exist_ok=False)  # generate an error if the directory already exists

                print(message1)
                print(message2)
                session_log.info(message1)
                session_log.info(message2)

            if not path.is_dir():
                raise RuntimeError('The supplied path: "{}" is not a directory. Exiting.'.format(path))
        else:
            name = '{}---{}'.format(default_name, timestamp_string)

            # Determine whether or not to save the the directory under the project root.
            if not save_in_project_dir:
                path = Path(Path.cwd(), name)
            else:
                path = Path(self.project_dir, name)

            message3 = 'Default {} directory created: "{}".'.format(default_name, path)
            try:
                path.mkdir(parents=True, exist_ok=False)

                print(message3)
                session_log.info(message3)
            except FileExistsError:
                raise RuntimeError('Could not create default {} path: "{}". Directory already exists.'.format(default_name, path))
        return path


    def _validate_random_seed(self):
        """Generate a random seed if one is not provided."""
        if self.random_seed == None:
            seed = int(datetime.now().timestamp())
            self.random_seed = seed


    def _validate_limits(self):
        """The classification algorithm has a very specific limits requirement. Determine whether the parameters
        passed meet that criterion."""
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


    def _validate_nice_level(self):
        """Determine whether the nice level is valid."""
        if self.nice_level > 20 or self.nice_level < -20:
            raise RuntimeError('The nice level can only be between -20 (highest) and 19 (lowest).')


    def _validate_plot_type(self):
        """Check the plot type."""
        if self.plot_type is None:
            self.plot_type = 'png'
        else:
            # Tiff can be supported, but requires the installation of Pillow, the replacement for the obsolete PIL package.
            allowed = 'png,pdf,svg,eps'.split(',')
            if self.plot_type.lower() not in allowed:
                raise RuntimeError('Unsupported plot extension: "{}". Try one of: "{}".'.format(self.plot_type, ', '.join(allowed)))


    def _validate_directories(self):
        """Check for the existence of the `project_directory` as well as the `work` and `logs` directories.
        Create the latter two if not found."""
        if self.project_dir is None:
            raise RuntimeError('No `project_directory` has been supplied in the settings. Exiting.')
        else:
            self._validate_project_directory()

        # Check for work and log paths existence. Create if not found.
        now_str = self._now_timestamp()

        # This is an implicit else for both `if` conditions.
        self.work_dir = str(self._create_directory_if_not_exists(self.work_dir, 'work', now_str))
        self.logs_dir = str(self._create_directory_if_not_exists(self.logs_dir, 'logs', now_str))


    def _validate_session_name(self):
        """Check the session name is non-empty and a string."""
        if self.session_name is None:
            raise RuntimeError('Session name cannot be `None`. Exiting.')

        if type(self.session_name) is not str:
            raise RuntimeError('The session name must be a string. Exiting.')

        if type(self.session_name) is str and len(self.session_name.strip()) == 0:
            raise RuntimeError('The session name cannot be empty, or contain only whitespace. Exiting.')


    def _validate_num_processes(self):
        """Validate the number of requested processes, and correct the number if too large to avoid
        performance degradation."""
        available_processes = mp.cpu_count()

        if self.num_processes is None:
            self.num_processes = available_processes

        if type(self.num_processes) is not int:
            raise RuntimeError('Only integer processes can be used.')

        if self.num_processes > available_processes:
            self.num_processes = available_processes
            raise RuntimeWarning('Warning: requesting more processes than available virtual CPUs. Performance may be degraded.')


    def _validate_filename_format(self):
        """Check the filename format and whether it contains the required `{well_name}` format string."""
        if type(self.filename_format) is not str:
            raise RuntimeError('The `filename_format` must be a string. Exiting.')
        else:
            if '{well_name}' not in self.filename_format:
                raise RuntimeError('The `filename_format` must contain the template variable: "{well_name}".')


    def _validate_project_directory(self):
        """Determine whether the project directory is valid."""
        path = Path(self.project_dir).expanduser().absolute()
        if not path.exists():
            raise RuntimeError('The `project_directory` does not exist. Exiting.')

        if not path.is_dir():
            raise RuntimeError('The path supplied for `project_directory` is not a directory. Exiting.')
        self.project_dir = str(path)


    def _validate_move_to_pdir(self):
        """Determine whether the option to move the generated plots into the `project_directory` after
        analysis is valid."""
        mtype = type(self.move_to_pdir)
        if mtype is not bool:
            raise RuntimeError('The `move_to_project_directory` cannot be `{}`. Please use a boolean value.'.format(mtype))


    def _validate_wells_filename(self):
        """Check whether or not the `wells_filename` is valid, and exists."""
        if self.wells_filename is None:
            raise RuntimeError('The `wells_filename` has not been set. Exiting.')

        updated_path = Path(self.project_dir).joinpath(self.wells_filename).absolute()  # `project_dir` is already expanded here.
        if not updated_path.exists():
            raise RuntimeError('The `wells_filename` ("{}") does not exist under the project path: "{}". Exiting.'.format(
                self.wells_filename,
                self.project_dir
            ))

        if not updated_path.is_file():
            raise RuntimeError('The `wells_filename` ("{}") is not a file. Exiting.'.format(str(updated_path)))

        self.wells_filename = str(updated_path)


    def _populate_bins_variables(self):
        """Using the set parameter values within the config object, define the `numpy` array which
        contains the limits for use in the classification algorithm."""
        self.num_conc_bins = int((self.high_conc - self.low_conc)/self.conc_bin_width) + 1
        self.num_fret_bins = int((self.high_fret - self.low_fret)/self.fret_bin_width) + 1

        self.conc_bins = np.linspace(self.low_conc, self.high_conc, self.num_conc_bins)
        self.fret_bins = np.linspace(self.low_fret, self.high_fret, self.num_fret_bins)
        self.fg_conc_edges = np.linspace(self.low_conc, self.high_conc, self.number_of_bins_xy + 1)
        self.fg_fret_edges = np.linspace(self.low_fret, self.high_fret, self.number_of_bins_xy + 1)


    def _validate_config(self, yaml_config):
        """Set the values for the internal parameters of the config object
        based on those parsed / extracted from an input YAML object (usually
        a dictionary). Finally, validate those values and auto-populate the
        relevant variables where necessary."""
        c = yaml_config.copy()

        # Job control parameters
        self.session_name       = c['session_name']
        self.random_seed        = c['random_seed']
        self.num_processes      = c['num_processes']
        self.nice_level         = c['nice_level']

        # Algorithm parameters
        self.low_conc_cutoff    = c['low_conc_cutoff']
        self.high_conc_cutoff   = c['high_conc_cutoff']
        self.low_conc           = c['low_conc']
        self.high_conc          = c['high_conc']
        self.low_fret           = c['low_fret']
        self.high_fret          = c['high_fret']
        self.conc_bin_width     = c['conc_bin_width']
        self.fret_bin_width     = c['fret_bin_width']
        self.number_of_bins_xy  = c['number_of_bins']
        self.min_measurements   = c['minimum_required_measurements']

        # Important files and directories
        self.project_dir        = c['project_directory']
        self.work_dir           = c['work_directory']
        self.logs_dir           = c['logs_directory']
        self.filename_format    = c['filename_format']
        self.wells_filename     = c['wells_filename']
        self.move_to_pdir       = c['move_to_project_directory']

        # Plotting options
        self.plot_gaussian      = c['plot_gaussian_fits']
        self.plot_logistic      = c['plot_logistic_fits']
        self.plot_fine_grids    = c['plot_2d_histograms']
        self.plot_rsquared      = c['plot_rsquared_fits']
        self.plot_skipped       = c['plot_skipped_2d_histograms']
        self.fine_grid_xlim     = c['xlim_2d_histogram']
        self.fine_grid_ylim     = c['ylim_2d_histogram']
        self.plot_type          = c['plot_type']

        self._populate_bins_variables()
        self.validate()


    def load_config_from_settings(self, yaml_config):
        """This is another convenience function which allows the population and validation of
        the config object directly from a parsed YAML config (often a dictionary)."""
        self._validate_config(yaml_config)


    def load_config_from_file(self, settings_filename):
        """This is a convenience function to allow the population and validation of the config
        object using the path to an input YAML file."""
        yaml_config = load_settings(settings_filename)
        self._validate_config(yaml_config)


    def validate(self):
        """Check all the parameters, and set them to appropriate defaults where necessary."""
        # Validate the session name
        self._validate_session_name()

        # Check for project, work, and logs directories. Create work and log if not found.
        self._validate_directories()

        # Check whether moving to the `project_directory` is valid.
        self._validate_move_to_pdir()

        # Check the conc and fret limits.
        self._validate_limits()

        # Check / set the random variable (if Null).
        self._validate_random_seed()

        # Check and validate the values / limits of the important variables.
        self._check_if_zero('min_measurements')
        self._check_if_zero('num_conc_bins')
        self._check_if_zero('num_fret_bins')
        self._check_if_zero('number_of_bins_xy')

        self._check_if_zero('high_conc')
        self._check_if_zero('conc_bin_width')
        self._check_if_zero('fret_bin_width')

        self._check_if_null('conc_bins')
        self._check_if_null('fret_bins')
        self._check_if_null('fg_conc_edges')
        self._check_if_null('fg_fret_edges')
        self._check_if_null('wells_filename')

        self._check_limits_var('fine_grid_xlim')
        self._check_limits_var('fine_grid_ylim')

        # Check that the nice level is within the allowed limits (-20 to 19)
        self._validate_nice_level()

        # Check that the number of requested processes is valid. Set a valid number accordingly.
        self._validate_num_processes()

        # Check that the filename_format is not Null.
        self._validate_filename_format()

        # Apply a path fix for the `wells_filename` if a relative path is used.
        self._validate_wells_filename()

        # Check for valid plot types.
        self._validate_plot_type()


    def __repr__(self):
        """Generate a string representation of the object and its parameter values for debugging
        purposes. For e.g. if our `settings_filename` is `config.yaml` we can do:

            >>> from damfret_classifier.config import Config
            >>> config = Config('config.yaml')
            >>> config_str = str(repr(config))
            >>> other_config = Config('config.yaml')
            >>> other_config_str = str(repr(other_config))
            >>> config == other_config  # False since different memory locations
            >>> config_str == other_config_str  # True

        Amongst other things.
        """

        # Build the list of parameters contained with the object. These will be used to select
        # their values while building the string representation of the object.
        params = 'settings_filename,session_name,random_seed,num_processes,nice_level'.split(',')
        params += 'low_conc,high_conc,low_fret,high_fret'.split(',')
        params += 'low_conc_cutoff,high_conc_cutoff,conc_bin_width,fret_bin_width'.split(',')
        params += 'number_of_bins_xy,min_measurements,move_to_pdir'.split(',')
        params += 'project_dir,work_dir,logs_dir,filename_format,wells_filename'.split(',')
        params += 'plot_gaussian,plot_logistic,plot_fine_grids,plot_rsquared'.split(',')
        params += 'plot_skipped,fine_grid_xlim,fine_grid_ylim,plot_type'.split(',')

        # Now we obtain the values, and populate them as key / value pairs which will
        # be appended and inserted into the string representation.
        #
        # The benefit of this approach is that we can also create objects directly
        # from the string.
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
        """Output a string representation of the object. Useful for printing."""
        return str(repr(self))
