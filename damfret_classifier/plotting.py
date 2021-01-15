import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


# Configure the plotting to generate PDFs which are editable via Illustrator.
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial']
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42


__all__ = ['plot_gaussian_fits', 'plot_logistic_fits', 'generate_fine_grid_plots']


# savepath, filename, norm_fret_hist, x, y1, y2, average, conc_slice, frac_nucleated, slice_number
def plot_gaussian_fits(plot_params):
    savepath        = plot_params['savepath']
    data_file       = plot_params['data_file']
    norm_fret_hist  = plot_params['norm_fret_hist']
    x               = plot_params['x']
    y1              = plot_params['y1']
    y2              = plot_params['y2']
    average         = plot_params['average']
    conc_slice      = plot_params['conc_slice']
    frac_nucleated  = plot_params['frac_nucleated']
    slice_number    = plot_params['slice_number']

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
    ax.set_title('Histogram of FRET values in the concentration slice %.2f - %.2f (nucleated = %f)\n(%s)' % (conc_slice[0], conc_slice[1], frac_nucleated, data_file))
    ax.legend()
    
    savename = os.path.join(savepath, '%s-slice-%d.png' % (data_file, slice_number))
    fig.savefig(savename)


# savepath, filename, conc_bin_centers, nucleated_fractions, y, rs, popt
def plot_logistic_fits(plot_params):
    savepath            = plot_params['savepath']
    data_file           = plot_params['data_file']
    conc_bin_centers    = plot_params['conc_bin_centers']
    nucleated_fractions = plot_params['nucleation_fractions']
    y                   = plot_params['y']
    r_squared           = plot_params['r_squared']
    popt                = plot_params['popt']

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111)
    
    ax.plot(conc_bin_centers, nucleated_fractions, 'go-', label='Nucleation Fraction')
    ax.plot(conc_bin_centers, y, 'r-', label='Logistic Function (a=1.00, b=%.2f, c=%.2f) [ $R^2 = %.3f$ ]' % (popt[0], popt[1], r_squared))
    
    ax.set_title('Nucleated Fractions across concentration slices ($R^2 = %.3f$)\n(%s)' % (r_squared, data_file))
    ax.set_ylim(0, 1.1)
    
    ax.axhline(1.0, linestyle='--')
    ax.set_ylabel('Fraction Nucleated')
    ax.set_xlabel('$log_{10}$ Concentration')
    ax.legend(loc=8)

    savename = os.path.join(savepath, 'logistic-fits---%s.png' % data_file)
    fig.savefig(savename)


# savepath, config, data, filename
def generate_fine_grid_plots(plot_params):
    savepath    = plot_params['savepath']
    config      = plot_params['config']
    data_df     = plot_params['data_df']
    data_file   = plot_params['data_file']

    hist, _xedges, _yedges = np.histogram2d(data_df['concentration'], data_df['FRET'], bins=(config.fg_conc_bins, config.fg_fret_bins))
    hist[hist == 0] = None

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, aspect='equal')
    im = ax.imshow(np.rot90(hist), cmap='jet')

    ax.set_title(data_file)
    ax.set_xlabel('$log_{10}$ Concentration')
    ax.set_ylabel('AmFRET/Concentration')
    ax.set_xticks(np.linspace(0, 300, 6))
    ax.set_xticklabels(['{:.2f}'.format(x) for x in np.linspace(config.low_conc, config.high_conc, 6)])
    ax.set_yticks(np.linspace(0, 300, 5))
    ax.set_yticklabels(['{:.2f}'.format(y) for y in np.linspace(config.high_fret, config.low_fret, 5)])
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='3%', pad=0.2)
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.set_ylabel('Counts', labelpad=30, rotation=270, fontsize=20)

    savename = os.path.join(savepath, '{data_filename}.png'.format(data_filename=data_file))
    plt.savefig(savename)


# ---------------------------------------------------------------------------------------------------------------------


# See this magnificent link: https://stackoverflow.com/questions/42683287/python-numpy-shannon-entropy-array
# Also see the file: `entropy_calculations.py` in the Scratch folder `Analysis_For_Ammon`.
def shannon_entropy(histogram2d):
    prob_distribution = histogram2d / histogram2d.sum()

    log_p = np.log(prob_distribution)
    log_p[log_p == -np.inf] = 0  # remove the infs
    entropy = -np.sum(prob_distribution * log_p)
    return entropy


def analyze_shannon_entropy(constructs_df, genes, savename, description, external_legend=False):
    markers = '^s*odx'

    ylolim = -1.0
    yuplim = 1.0
    xlolim = 0
    xuplim = 5

    bins = 10**np.log10(np.logspace(1, 4, 10))
    
    fig = plt.figure(figsize=(16, 8))
    axes = fig.subplots(1, 2)
    plt.subplots_adjust(wspace=0.3)
    all_max_dydx = list()
    index = 0
    replicates = '2,1,1,3,2,3'.split(',')
    for well_file, replicate_str in zip(constructs_df, replicates):
        replicate = int(replicate_str)
        data = constructs_df[well_file]
        #data = df.rename(columns={'concentration':'acceptor', 'FRET':'damfret'})
        #print(data)
        # clamp the data
        data = data[data['damfret'] <= yuplim]
        data = data[data['damfret'] >= ylolim]

        entropies = list()

        for num_bins in bins:
            xbins = np.linspace(xlolim, xuplim, int(num_bins)+1)
            ybins = np.linspace(ylolim, yuplim, int(num_bins)+1)
            hist, xedges, yedges = np.histogram2d(data['acceptor'], data['damfret'], bins=(xbins, ybins))
            
            # now check the Shannon Entropy
            entropy = shannon_entropy(hist)
            entropies.append(entropy)
            print(entropy, num_bins)
        
        marker = markers[index]
        bin_widths = (xuplim - xlolim)/np.array(bins)
        axes[0].semilogx(bins, entropies, marker, linestyle=':', label='%s replicate %d' % (genes[index], replicate))
        
        xx = np.log10(bin_widths)
        yy = entropies[::]
        dydx = np.diff(yy)/np.diff(xx)
        
        # See: https://stackoverflow.com/a/26042315/866930
        dydx = np.gradient(entropies, xx)

        df = pd.DataFrame()
        df['bins'] = bins
        df['bin-width'] = bin_widths
        df['entropy'] = entropies
        df['delta-entropy'] = np.abs(dydx)
        #df.to_fwf(os.path.join(ENTROPIES_DIR, '%03d___%d.tsv' % (construct, replicate)))

        if not external_legend:
            axes[1].semilogx(bins, np.abs(dydx), marker, linestyle=':', label='%s replicate %d' % (genes[index], replicate))
        else:
            axes[1].semilogx(bins, np.abs(dydx), marker, linestyle=':', label='%s replicate %d (%s)' % (genes[index], replicate, description[index]))
        axes[0].tick_params(axis='x', labelsize=16)
        axes[0].tick_params(axis='y', labelsize=16)
        axes[1].tick_params(axis='x', labelsize=16)
        axes[1].tick_params(axis='y', labelsize=16)
        max_dydx_index = np.where(np.abs(dydx) == max(np.abs(dydx)))[0]
        all_max_dydx.append(bins[max_dydx_index])
        axes[1].axvline(bins[max_dydx_index], linestyle='-', zorder=-1, linewidth=0.5, color='grey')
        index += 1
        

    if not external_legend:
        axes[0].set_title('Shannon Entropy (%s)' % description, fontsize=20)
    else:
        axes[0].set_title('Shannon Entropy as a function of Grid Size', fontsize=20)
    axes[0].set_xlabel('Grid Size', fontsize=16)
    axes[0].set_ylabel('$S$', fontsize=16)

    lgd = None
    if not external_legend:
        axes[1].set_title('Change in Shannon Entropy (%s)' % description, fontsize=20)
    else:
        axes[1].set_title('Change in Shannon Entropy with respect to Grid Size', fontsize=20)
    axes[1].set_xlabel('Grid Size', fontsize=16)
    axes[1].set_ylabel(r'$|\Delta S|$', fontsize=16)

    if not external_legend:
        axes[0].legend()
        axes[1].legend()
    else:
        lgd = axes[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 16})

    savepath = os.path.join('%s.pdf' % savename)
    if not external_legend:
        fig.savefig(savepath)
    else:
        fig.savefig(savepath, bbox_extra_artists=(lgd,))