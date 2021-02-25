import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from damfret_classifier.utils import create_directory_if_not_exist, shannon_entropy
from mpl_toolkits.axes_grid1 import make_axes_locatable


# Configure the plotting to generate PDFs which are editable via Illustrator.
mpl.rcParams['font.family']     = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial']
mpl.rcParams['pdf.fonttype']    = 42
mpl.rcParams['ps.fonttype']     = 42


__all__ = ['plot_gaussian_fits', 'plot_logistic_fits', 'plot_linear_rsquared_fit', 'plot_fine_grid_profiles',
           'analyze_shannon_entropy']


def plot_gaussian_fits(plot_params):
    """This function plots the Double-Gaussian fits to the 1D FRET histogram extracted within a concentration slice. 
    It should be noted that for a given data file the center of the second gaussian will remain fixed, as its fraction
    filled is used as a proxy for nucleation. As the number of parameters required is unwiedly, a dictionary 
    `plot_params` is used to pass all the requisite variables.

    Upon successful running of this program, a plot corresponding to the prescribed plot type will be generated.
    """
    savepath        = plot_params['savepath']
    well_name       = plot_params['well_name']
    norm_fret_hist  = plot_params['norm_fret_hist']
    x               = plot_params['x']
    y1              = plot_params['y1']
    y2              = plot_params['y2']
    average         = plot_params['average']
    conc_slice      = plot_params['conc_slice']
    frac_nucleated  = plot_params['frac_nucleated']
    slice_number    = plot_params['slice_number']
    extension       = plot_params['plot_type']

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111)
    
    ax.plot(x, norm_fret_hist, 'g-', linewidth=3.0, label='Normalized FRET counts')
    ax.plot(x, y1+y2, 'm-', label='Combined Gaussian', linewidth=2.0)
    ax.plot(x, y1, 'r-', label='Gaussian 1 (center = 0)', linewidth=2.0)
    ax.plot(x, y2, 'b-', label='Gaussian 2 (center = %.3f)' % average, linewidth=2.0)
    
    ax.axvline(0, linestyle=':', color='grey', label='Gaussian 1 center', zorder=-1, linewidth=1.0)
    ax.axvline(average, linestyle='--', color='grey', label='Gaussian 2 center', zorder=-1, linewidth=1.0)
    
    ax.set_xlabel('FRET')
    ax.set_ylabel('Normalized Frequency')
    ax.set_title('Histogram of FRET values in the concentration slice %.2f - %.2f (nucleated = %f)\n(%s)' % (conc_slice[0], conc_slice[1], frac_nucleated, well_name))
    ax.legend()
    
    savename = os.path.join(savepath, '{well_name}---slice-{slice_num:02d}.{ext}'.format(well_name=well_name, slice_num=slice_number, ext=extension))
    fig.savefig(savename)
    plt.close(fig)


def plot_logistic_fits(plot_params):
    """This function plots the logistic function fit to the nucleated fractions as determined by the parameters of the double-Gaussian fit. 
    As the number of parameters required is unwiedly, a dictionary `plot_params` is used to pass all the requisite variables.

    Upon successful running of this program, a plot corresponding to the prescribed plot type will be generated.
    """
    savepath            = plot_params['savepath']
    well_name           = plot_params['well_name']
    conc_bin_centers    = plot_params['conc_bin_centers']
    nucleated_fractions = plot_params['nucleated_fractions']
    y                   = plot_params['logistic_y']
    r_squared           = plot_params['r_squared']
    popt                = plot_params['popt']
    extension           = plot_params['plot_type']

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111)
    
    ax.plot(conc_bin_centers, nucleated_fractions, 'go-', label='Nucleation Fraction')
    ax.plot(conc_bin_centers, y, 'r-', label='Logistic Function (a=1.00, b=%.2f, c=%.2f) [ $R^2 = %.3f$ ]' % (popt[0], popt[1], r_squared))
    
    ax.set_title('Nucleated Fractions across concentration slices ($R^2 = %.3f$)\n(%s)' % (r_squared, well_name))
    ax.set_ylim(0, 1.1)
    
    ax.axhline(1.0, linestyle='--')
    ax.set_ylabel('Fraction Nucleated')
    ax.set_xlabel('$log_{10}$ Concentration')
    ax.legend()

    savename = os.path.join(savepath, '{well_name}---logistic-fit.{ext}'.format(well_name=well_name, ext=extension))
    fig.savefig(savename)
    plt.close(fig)


def plot_linear_rsquared_fit(plot_params):
    """This function plots the linear fit to the R^2 values extracted from the nucleated fraction slice calculations. Namely, it fits a linear
    function to the R^2 values of the fits to the double-Gaussian function. As the number of parameters required is unwiedly, a dictionary 
    `plot_params` is used to pass all the requisite variables.

    Upon successful running of this program, a plot corresponding to the prescribed plot type will be generated.
    """
    savepath                        = plot_params['savepath']
    conc_bin_centers                = plot_params['conc_bin_centers']
    nucleated_fraction_r_squared    = plot_params['r_squared']
    linear_func_data                = plot_params['linear_func_data']
    well_name                       = plot_params['well_name']
    linear_r_squared                = plot_params['r']
    popt                            = plot_params['popt']
    extension                       = plot_params['plot_type']

    fig = plt.figure(figsize=(9,8))
    ax = fig.add_subplot(111)
    ax.plot(conc_bin_centers, nucleated_fraction_r_squared, '.', ms=10, label='$R^2$')
    ax.plot(conc_bin_centers, linear_func_data, 'r-', label='linear fit (a=%.3f, b=%.3f) [ $R^2$ = %.3f ]' % (popt[0], popt[1], linear_r_squared))
    ax.set_title(well_name)
    ax.set_xlabel('$log_{10}$ Concentration')
    ax.set_ylabel('$R^2$ value of the Guass2 fit on sliced AmFRET data')
    ax.legend()

    savename = os.path.join(savepath, '{well_name}---rsquared-fit.{ext}'.format(well_name=well_name, ext=extension))
    fig.savefig(savename)
    plt.close(fig)


def plot_fine_grid_profiles(plot_params):
    """This function plots fine grid profiles of the dataset in question. As the number of parameters required is unwiedly,
    a dictionary `plot_params` is used to pass all the requisite variables.

    Upon successful running of this program, a plot corresponding to the prescribed plot type will be generated.
    """
    savepath        = plot_params['savepath']
    well_name       = plot_params['well_name']
    fg_conc_edges   = plot_params['fg_conc_edges']
    fg_fret_edges   = plot_params['fg_fret_edges']
    data_df         = plot_params['data_df']
    fine_grid_xlim  = plot_params['fine_grid_xlim']
    fine_grid_ylim  = plot_params['fine_grid_ylim']
    extension       = plot_params['plot_type']

    hist, _xedges, _yedges = np.histogram2d(data_df['concentration'], data_df['damfret'], bins=(fg_conc_edges, fg_fret_edges))
    z = hist.transpose()
    z = np.ma.masked_array(z, z == 0)

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111)

    im = ax.pcolorfast(_xedges, _yedges, z, cmap='jet')
    im.set_clip_path(ax.patch)

    ax.set_title(well_name)
    ax.set_xlabel('$log_{10}$ Concentration')
    ax.set_ylabel('AmFRET/Concentration')
    
    if fine_grid_xlim is not None and fine_grid_ylim is not None:
        xlow, xhigh = fine_grid_xlim
        ylow, yhigh = fine_grid_ylim
        ax.set_xlim(xlow, xhigh)
        ax.set_ylim(ylow, yhigh)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='3%', pad=0.2)
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.set_ylabel('Counts', labelpad=30, rotation=270, fontsize=20)

    savename = os.path.join(savepath, '{well_name}---fine-grid.{ext}'.format(well_name=well_name, ext=extension))
    fig.savefig(savename)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------------------------


def analyze_shannon_entropy(config, constructs_df, genes, replicates, descriptions, savename):
    """This function allows the user to analyze the Shannon Entropy of a given subset of the data. In so doing, it 
    provides an avenue for the user to determine the ideal grid size and the minimum number of points to use as
    a selection criteria. The amount determined in the manuscript is 20K. However, as that number was specific to
    the dataset used, a different number may be required for other datasets, hence the existence of this function.

    Upon successful running of this program, it will produce a plot image of 2 panels which will show the Shannon
    Entropy as a function of grid size and the absolute change in Shannon Entropy also as a function of grid size.

    The peak of the second panel is an indicator as to the optimal grid size to be used for that dataset.

    @param config (Config):             A `Config` instance with all the requisite parameters for analysis.
    @param constructs_df (OrderedDict): An `OrderedDict` containing a mapping of well filenames to their `DataFrames`.
    @param genes (list):                The corresponding gene names to the well filenames.
    @param replicates (list):           The replicate numbers of the genes.
    @param descriptions (OrderedDict):  A description of the datasets used - e.g. 10K, 20K, 30K.
    @param savename (str):              The name of the plot to save.
    """
    markers = '^s*odxPvXh'
    if len(genes) > len(markers):
        raise RuntimeError('Only a maximum of {} replicates can be plotted simulataneously.'.format(len(markers)))

    ylolim = config.low_fret
    yuplim = config.high_fret
    xlolim = config.low_conc
    xuplim = config.high_conc

    bins = 10**np.log10(np.logspace(1, 4, 10))
    
    fig = plt.figure(figsize=(16, 8))
    axes = fig.subplots(1, 2)
    plt.subplots_adjust(wspace=0.3)
    all_max_dydx = list()
    index = 0
    for well_name, replicate_str in zip(constructs_df, replicates):
        replicate = int(replicate_str)
        data = constructs_df[well_name]
        
        # clamp the data
        data = data[data['damfret'] <= yuplim]
        data = data[data['damfret'] >= ylolim]

        entropies = list()

        for num_bins in bins:
            xbins = np.linspace(xlolim, xuplim, int(num_bins)+1)
            ybins = np.linspace(ylolim, yuplim, int(num_bins)+1)
            hist, _xedges, _yedges = np.histogram2d(data['acceptor'], data['damfret'], bins=(xbins, ybins))
            
            # now check the Shannon Entropy
            entropy = shannon_entropy(hist)
            entropies.append(entropy)
        
        marker = markers[index]
        bin_widths = (xuplim - xlolim)/np.array(bins)
        axes[0].semilogx(bins, entropies, marker, linestyle=':', label='%s replicate %d (%s)' % (genes[index], replicate, descriptions[index]))
        
        xx = np.log10(bin_widths)
        yy = entropies[::]
        dydx = np.diff(yy)/np.diff(xx)
        
        # See: https://stackoverflow.com/a/26042315/866930
        dydx = np.gradient(entropies, xx)

        axes[1].semilogx(bins, np.abs(dydx), marker, linestyle=':', label='%s replicate %d (%s)' % (genes[index], replicate, descriptions[index]))
        axes[0].tick_params(axis='x', labelsize=16)
        axes[0].tick_params(axis='y', labelsize=16)
        axes[1].tick_params(axis='x', labelsize=16)
        axes[1].tick_params(axis='y', labelsize=16)
        max_dydx_index = np.where(np.abs(dydx) == max(np.abs(dydx)))[0]
        all_max_dydx.append(bins[max_dydx_index])
        axes[1].axvline(bins[max_dydx_index], linestyle='-', zorder=-1, linewidth=0.5, color='grey')
        index += 1

    axes[0].set_title('Shannon Entropy as a function of Grid Size', fontsize=20)
    axes[0].set_xlabel('Grid Size', fontsize=16)
    axes[0].set_ylabel('$S$', fontsize=16)

    lgd = None
    axes[1].set_title('Change in Shannon Entropy with respect to Grid Size', fontsize=20)
    axes[1].set_xlabel('Grid Size', fontsize=16)
    axes[1].set_ylabel(r'$|\Delta S|$', fontsize=16)

    lgd = axes[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 16})

    savepath = os.path.join('{savename}.{ext}'.format(savename=savename, ext=config.plot_type))
    fig.savefig(savepath, bbox_extra_artists=(lgd,))
    plt.close(fig)
