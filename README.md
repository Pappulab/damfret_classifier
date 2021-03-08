# Overview

This Python 3 package provides functionality to analyze and classify DAmFRET data by mechanistic inferences based on a supervised learning algorithm. This algorithm is an implementation of the one documented in the Posey et al., 2021 paper: ["Mechanistic inferences from analysis of measurements of protein phase transitions in live cells"](https://www.biorxiv.org/content/10.1101/2020.11.04.369017v2). **This package is currently in production pre-release, so please exercise care with its use.**

Six possible phase transition states are hypothesized to exist and are identified with this method, which also outputs confidence scores (0 to 1 which characterize low and high confidence, respectively):

1. Assembly at all concentrations (**Black Class**)
2. No assembly at all concentrations (**Blue Class**)
3. Continuous Transition (**Red Class**)
4. Discontinuous Transition (**Green Class**)
5. Higher Order State (**Magenta Class**)
6. Incomplete Transition (**Yellow Class**)


# Table Of Contents

- [Installation](#installation)
  * [Dependencies](#dependencies)

- [Script Usage](#script-usage)
  * [DAmFRET Classify](#damfret-classify)
   	+ [DAmFRET Classify Output](#damfret-classify-output)
	+ [Estimated Runtime](#estimated-runtime)
  * [TSV Comparator](#tsv-comparator)

- [API usage](#api-usage)
- [Citation](#citation)


# Installation

The package can be installed via: `pip install git+https://github.com/pappulab/damfret_classifier.git`. If previous versions have been installed, it is recommended that those are removed first: i.e. `pip uninstall damfret_classifier`.

## Dependencies

This package requires **Python 3.5+** as well as the following packages as dependencies, which are installed as needed when installation is performed:

```
matplotlib>=3.3
numpy>=1.19
pandas>=1.2
PyYAML>=5.4
scipy>=1.5
tabulate>=0.8
```

# Script Usage

Upon installation, two scripts are provided and made available in the user's `$PATH`: `damfret_classify` and `tsv-comparator`. The first script is meant to be used for analyzing data while the second is for comparison of the results across different runs.

## DAmFRET Classify

The `damfret_classify` script adapts the API into a functional tool which can be used to analyze various DAmFRET datasets. By default, the script supports 2 flags (in addition to the traditional help flag, `-h` or `--help`):

1. `-g` or `--generate-default-config`: this option generates a default [YAML](https://en.wikipedia.org/wiki/YAML) config file, `config.yaml`, that the `damfret_classify` script uses. It contains several well-documented options that the user can introspect and adapt for their project.

2. `-c` or `--config`: this option allows the user to provide the location and name of the YAML configuration file to use. If the user has a YAML config file named `config.yaml` in the current directory where the script is run, this flag does not have to be explicitly called as the script will use that configuration file. The basis for this behavior is to enable easier use and improved scriptability.

It should be noted that relative paths are supported in the config options which require filepaths or directory locations. And, if the config's `project_directory` option contains several sub-directories of individual data - i.e. those which contain files with the same extension as `filename_format` in the config - those are enumerated for analysis. Only sub-directories which start with `raw_data` from the current path where the script is run are ignored. 

In such a case, a copy of the current `config.yaml` is made to that sub-directory named as `damfret_classifier_settings.yaml`, wherein the `project_directory` and any other relevant options have been updated.

For e.g., if no `wells_filename` option is provided in the YAML config, each directory of interest is examined for well files and a default CSV file called `wells.csv` is generated and saved to the sub-directory's location. In short, this configuration allows for multiple datasets - even those with the same well names - to be stored under the same `project_directory` and be analyzed in succession.

To provide a means for reproducibility and comparison of determined classifications, one can set the random seed. If several sub-directories are found within the `project_directory`, the same random seed is used for the analysis of the data as a means of uniform comparison. If no random seed is provided, one will be generated and used.

### DAmFRET Classify Output

While the script runs, a **session log** is generated and displayed to the user which contains messages and information about the subsequent analysis for the current sub-directory. Summaries are provided for the concentration limits of each well file, which well files were identified as having a given phase-transition class, and more. That **session log** is saved with the name `session---<YYYY-mm-DD_HH-MM-SS>.log` to the provided `work_directory`. 

If no path of the work directory is provided, a default one is created with the format: `work---<YYYY-mm-DD_HH-MM-SS>`. If plots are requested, these are first saved to the `work_directory` and only moved into the `project_directory` if requested per the YAML config options. Once the analysis completes, the results summary is saved as `parameters.{csv,tsv}` in the `work_directory`. If any well files were skipped due to the number of cell measurements, well file non-existence, and well files containing data whose concentration exceeded the provided limits, those are saved as `skipped.{csv,tsv}` in the `work_directory`.

The log file for each individual well is stored separately to allow for easier examination. The individual logs are saved to the `logs_directory` in the YAML config. Similar to the `work_directory`, if no location is provided, the logs are saved to the directory containing the dataset for analysis in the format: `logs---<YYYY-mm-DD_HH-MM-SS>`.

The final `parameters.csv` and `parameters.tsv` files are identical, although the TSV file is perhaps easier to read and intropect. Documented therein are relevant parameters to help the user determine the quality of the classification and other related parameters using the devised classifcation algorithm. TSV data can be easily read via the `pandas` library:

```{python}
import pandas as pd
filename = 'parameters.tsv'
df = pd.read_csv(filename, sep='\\s+')  # i.e. use multiple spaces as our separator.
```

### Estimated Runtime

Using the [full experimental dataset from the DAmFRET manuscript](https://www.stowers.org/research/publications/libpb-1594), which is comprised of 380 well files (i.e. roughly 1 plate), an analysis takes ~ 2.65 minutes using 16 processes (Xeon W5580) if all intermediate plots are generated. This translates to an average of about ~ 6.7 seconds per well file / process. Your runtime may vary depending on your hardware configuration.


## TSV Comparator

Sometimes it is necessary to compare different runs and their resulting `parameters.tsv` (and CSV) files. The script `tsv-comparator` provides an easy interface to quickly introspect and compare such runs, e.g. cases where different random seeds were employed in the analysis of the same dataset. This comes in handy when testing and developing different 


# API Usage

In Progress.

# Citation

This package was used for analysis of the real and synthetic data presented in the  manuscript *"Mechanistic inferences from analysis of measurements of protein phase transitions in live cells"*:


```{bibtex}
@article{posey2020mechanistic,
	title={Mechanistic inferences from analysis of measurements of protein phase transitions in live cells},
	author={Posey, Ammon E and Ruff, Kiersten M and Lalmansingh, Jared M and Kandola, Tejbir S and Lange, Jeffrey J and Halfmann, Randal and Pappu, Rohit V},
	journal={bioRxiv},
	year={2020},
	publisher={Cold Spring Harbor Laboratory}
}
```

