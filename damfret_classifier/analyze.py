import os
import pandas as pd
from collections import OrderedDict
from tabulate import tabulate
from damfret_classifier.plotting import *
from damfret_classifier.utils import *
from damfret_classifier.config import Config


# Supremely useful answer to convert a DataFrame to a TSV file: https://stackoverflow.com/a/35974742/866930
def to_fwf(df, fname):
    content = tabulate(df.values.tolist(), list(df.columns), tablefmt="plain", floatfmt='.5f')
    open(fname, 'w').write(content)


pd.DataFrame.to_fwf = to_fwf


# ---------------------------------------------------------------------------------------------------------------------


def clamp_data(config, data):
    data = data.dropna()  # remove NaN values
    data = data[data['concentration'] >= config.lower_conc]
    data = data[data['concentration'] <= config.higher_conc]
    return data


def slice_data(data, config):
    # now, go through the concentration slices
    conc_fret_averages = list()
    slices_histograms = OrderedDict()
    for low, high in zip(config.conc_bins[:-1], config.conc_bins[1:]):
        df = data.copy()
        df = df[df['concentration'] > low]
        df = df[df['concentration'] <= high]
        
        # histogram the fret data in the concencration slice
        if not df.empty:
            fret_hist, _edges = np.histogram(df['FRET'], bins=config.fret_bins)
            norm_fret_hist = fret_hist / fret_hist.sum()

            df = df[df['FRET'] > 0.05]
            if not df.empty:
                average_fret = df['FRET'].mean()
                conc_fret_averages.append(average_fret)

                key = (low, high)
                slices_histograms[key] = (df, norm_fret_hist)  # we'll use the norm hist for fitting
    return conc_fret_averages, slices_histograms


def calculate_nucleated_fractions(config, slices_histograms, average):
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
        #plot_gaussian_fits(filename, norm_fret_hist, x, y1, y2, average, conc_slice, frac_nucleated, slice_number)

        nucleated_fractions.append(frac_nucleated)
        conc_bin_center = conc_slice[0] + config.conc_bin_width/2.0

        conc_bin_centers.append(conc_bin_center)
        r_squared.append(rs)
        slice_number += 1
    return nucleated_fractions, conc_bin_centers, r_squared


def determine_class(df_points, fraction_above_csat, diff, fit_value, region_r_squared, r):
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


def classify_datasets(config, genes_table, data_directory, prefix):
    if not isinstance(config, Config):
        raise RuntimeWarning('A `damfret_classifier.Config` object is required.')

    # A lambda function is used for fitting as the parameters are dynamically set.
    # Typically defined functions i.e. using `def` is not mutable and only one reference
    # is populated - remaining calls are not updated with new values.
    logistic_func = lambda x, b, c: 1.0/(1.0+np.exp(-(x-b)/c))
    logistic_bounds = [[1, 0], [20, 2]]
    p0 = (5, 0.3)

    params = OrderedDict()
    params['gene'] = list()
    params['construct'] = list()
    params['replicate'] = list()
    params['well_file'] = list()
    params['counts'] = list()
    params['mean-fret'] = list()
    params['gauss2-loc'] = list()
    params['csat'] = list()
    params['csat-slope'] = list()
    params['linear-r2'] = list()
    params['max-gauss-r2'] = list()
    params['min-r2-region'] = list()
    params['max-r2-region'] = list()
    params['min-abs-diff-r2-region'] = list()
    params['max-abs-diff-r2-region'] = list()
    params['frac-above-csat'] = list()
    params['color'] = list()
    params['score'] = list()

    counted = list()
    confidence_scores = OrderedDict()
    for _index, row in genes_table.iterrows():
        well = int(row.well_file[-2:])
        well_name = '%s%d' % (row.well_file[0], well)
        
        # compare against the actual data
        well_file = os.path.join(data_directory, '%s.fcs.csv' % well_name)
        filename = well_file[:]
        data, counts = read_original_data(well_file, config.low_conc_cutoff, config.high_conc_cutoff)
        counted.append(well_file)
        replicate = len(counted)
        if len(counted) == config.num_replicates:
            counted = list()
        
        params['counts'].append(counts) # NOT `len(data)` as it has already been pruned
        params['gene'].append(row.gene)
        params['well_file'].append(row.well_file)
        
        # remove extreme concentration values
        data = clamp_data(config, data)
        params['mean-fret'].append(data['FRET'].mean())
        
        # plot the fine-grid plots
        ###generate_fine_grid_plots(data, filename)
        conc_fret_averages, slices_histograms = slice_data(data, config)
        average = max(conc_fret_averages[-4:])  # previously from [-4:]
        print(average)
        print('CONC FRET AVERAGES: ', conc_fret_averages, conc_fret_averages[-4:])
        params['gauss2-loc'].append(average)
        nucleated_fractions, conc_bin_centers, r_squared = calculate_nucleated_fractions(config, slices_histograms, average)

        print()
        conc_bin_centers = np.array(conc_bin_centers)
        r_squared = np.array(r_squared)
        params['max-gauss-r2'].append(np.max(r_squared))
        
        # fit a linear function
        linear = lambda x, a, b: a*x + b

        popt, pconv = curve_fit(linear, conc_bin_centers, r_squared, bounds=[[-np.inf, -np.inf], [0, np.inf]])
        linear_func_data = linear(conc_bin_centers, *popt)
        _p, r = calculate_rsquared(linear, conc_bin_centers, r_squared, bounds=[[-np.inf, -np.inf], [0, np.inf]])

        # fig = plt.figure(figsize=(9,8))
        # ax = fig.add_subplot(111)
        # conc_bin_centers = conc_bin_centers[4:-3]
        # r_squared = r_squared[4:-3]
        # nucleated_fractions = nucleated_fractions[4:-3]
        # ax.plot(conc_bin_centers, r_squared, '.', ms=10, label='$R^2$')
        # ax.plot(conc_bin_centers, linear_func_data, 'r-', label='linear fit (a=%.3f, b=%.3f) [ $R^2$ = %.3f ]' % (popt[0], popt[1], r))
        # ax.set_title(filename.split('.')[0])
        # ax.set_xlabel('Log 10 Concentration')
        # ax.set_ylabel('$R^2$')
        # ax.legend()
        # fig.savefig('rsquared-%s.png' % filename.split('.')[0])

        # now, analyze the fraction that's nucleated to determine if the system is in one or two-states
        diff = nucleated_fractions[-1] - np.min(nucleated_fractions)
        if nucleated_fractions[-1] - np.min(nucleated_fractions) < 0.15:
            mean_fret = data['FRET'].mean()
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
            
            if mean_fret < 0.05:  # change to 0.1
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
        # plot_logistic_fits(filename, conc_bin_centers, nucleated_fractions, logistic_y, logistic_rs, popt)

        if saturation_conc <= config.lower_conc:
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
        
        print(filename)
        
        if saturation_conc < max(conc_bin_centers):
            region_indices = np.where((conc_bin_centers >= saturation_conc - 1.0) & (conc_bin_centers <= saturation_conc + 1.0))
            print('csat lesser', 9999)
        else:
            indices = list(range(len(conc_bin_centers)))
            region_indices = indices[-5:]
            print('csat greater')

        region_lower = np.where(conc_bin_centers >= saturation_conc)
        region_upper = np.where(conc_bin_centers <= saturation_conc)
        
        print(region_indices)
        print(region_lower, region_upper, 111)
        print(conc_bin_centers[region_lower], saturation_conc, conc_bin_centers[region_upper], 111)
        print(conc_bin_centers[region_indices], 222)
        print(saturation_conc, 123)

        # determine if there is a large change in the fit goodness in this region. And, determine the max change in
        # fit goodness around the saturation concentration.
        region_r_squared = r_squared[region_indices]
        print(region_r_squared, 234)
        fit_value = np.max(np.abs(np.diff(region_r_squared)))
        
        params['construct'].append(row.construct)
        params['replicate'].append(replicate)
        params['csat'].append(saturation_conc)
        params['csat-slope'].append(slope)
        params['linear-r2'].append(r)
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
        df = df[df['FRET'] > 0.05].dropna()

        total_points = len(data)
        df_points = len(df)
        fraction_above_csat = df_points/total_points
        params['frac-above-csat'].append(fraction_above_csat)
        
        color, score = determine_class(df_points, fraction_above_csat, diff, fit_value, region_r_squared, r)
        confidence_scores[filename] = (color, score)
        params['color'].append(color)
        params['score'].append(score)

    np.save('confidence-scores-real---%s.npy' % prefix, confidence_scores, allow_pickle=True)

    params = pd.DataFrame(params)
    params.to_fwf('parameters-real---%s.tsv' % prefix)
