class Parameters(object):
    """This class is a POD object (i.e. Plain Old Data object which is only comprised 
    of data elements); no methods. It contains the most important parameters calculated
    during processing of a given gene and its associated well_file data. These will
    eventually be placed in a Pandas DataFrame and saved as a TSV, which can then be
    used for subsequent analysis or validation.
    
    Attributes:

        gene (str)              (str) The name of the corresponding gene - e.g. EPL1.

        construct               (int) The construct number (arbitrary). Ranges from 1
                                to the number of constructs.
        
        replicate               (int) The replicate number of the current well file.
                                Ranges from 1 to the number of replicates
        
        well_file               (str) The name of the well_file (zero-padded). For
                                e.g. `I07`.
        
        counts                  (int) The number of points - i.e. cells - contained
                                in the file before processing. Used for validation
                                and identification.
        
        mean_fret               (float) The average of the FRET after the data has
                                removed extraneous points outside of the working limits.
                                See `damfret_classifier.analyze:clamp_data`.
        
        gauss2_loc              (float) The location of the center of the 2nd Gaussian used when
                                calculating the nucleated fractions. Used primarily for debugging
                                and algorithm fine-tuning.
        
        csat                    (float) The inferred location of the saturation concentration
                                via the algorithm. This parameter is used for subsequent
                                determination of the various classes.
        
        csat_slope              (float) The slope of the logistic function that has been fitted
                                to the nucleated fractions. Mainly used for debugging and fine
                                tuning of the algorithm.
        
        linear_r2               (float) The R^2 value for the linear function fit to the R^2
                                values of the Gauss2 function across each slice. Used primarily
                                for debugging and algorithm fine-tuning.
        
        max_gauss_r2            (float) The max of the R^2 values calculated for the fits of the
                                Gauss2 function. Used primarily for debugging and algorithm fine-tuning.
        
        min_r2_region           (float) The min of the R^2 values found within the region around
                                or close to the saturation concentration. Used for debugging and
                                fine-tuning the algorithm.
        
        max_r2_region           (float) The max of the R^2 values found within the region around
                                or close to the saturation concentration. Used for debugging and
                                fine-tuning the algorithm.
        
        min_abs_diff_r2_region  (float) The absolute value of the min n-th discrete difference.
                                Used primarly for debugging and fine-tuning the algorithm.
                                See: https://numpy.org/doc/stable/reference/generated/numpy.diff.html
        
        max_abs_diff_r2_region  (float) The absolute value of the max n-th discrete difference.
                                Used primarly for debugging and fine-tuning the algorithm.
                                See: https://numpy.org/doc/stable/reference/generated/numpy.diff.html
        
        color                   (str) The phase-separation classification. Possible values
                                are:
                                    black:      Assembled at all expression levels.
                                    blue:       No assembly at all expression levels.
                                    red:        Two-state continuous transition.
                                    green:      Two-state discontinuous transition.
                                    magenta:    Higher order state transition.
                                    yellow:     Infrequent transition.
        
        score                   (float) The confidence score of the color / class
                                assignment. The scores are ranged from 0 (least confident)
                                to 1 (most confident).
    """
    gene                    = 'N/A'
    construct               = 0
    replicate               = 0
    well_file               = 'N/A'
    counts                  = 0
    mean_fret               = 0.0
    gauss2_loc              = 0.0
    csat                    = 0.0
    csat_slope              = 0.0
    linear_r2               = 0.0
    max_gauss_r2            = 0.0
    min_r2_region           = 0.0
    max_r2_region           = 0.0
    min_abs_diff_r2_region  = 0.0
    max_abs_diff_r2_region  = 0.0
    color                   = 'N/A'
    score                   = 0.0