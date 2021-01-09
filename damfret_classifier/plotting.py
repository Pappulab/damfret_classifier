import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_gaussian_fits(savepath, filename, norm_fret_hist, x, y1, y2, average, conc_slice, frac_nucleated, slice_number):
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111)
    #plt.subplots_adjust(right=0.7)
    ax.plot(x, norm_fret_hist, 'g-', linewidth=3.0, label='Normalized FRET counts')
    ax.plot(x, y1+y2, 'm-', label='Combined Gaussian', linewidth=2.0)
    ax.plot(x, y1, 'r-', label='Gaussian 1 (center = 0)', linewidth=2.0)
    ax.plot(x, y2, 'b-', label='Gaussian 2 (center = %.3f)' % average, linewidth=2.0)
    ax.axvline(0, linestyle=':', color='grey', label='Gaussian 1 center', zorder=-1, linewidth=1.0)
    ax.axvline(average, linestyle='--', color='grey', label='Gaussian 2 center', zorder=-1, linewidth=1.0)
    ax.set_xlabel('FRET')
    ax.set_ylabel('Normalized Frequency')
    ax.set_title('Histogram of FRET values in the concentration slice %.2f - %.2f (nucleated = %f)\n(%s)' % (conc_slice[0], conc_slice[1], frac_nucleated, filename.split('.')[0]))
    ax.legend()
    
    savename = os.path.join(savepath, '%s-slice-%d.png' % (filename.split('.')[0], slice_number))
    fig.savefig(savename)


def plot_logistic_fits(savepath, filename, conc_bin_centers, nucleated_fractions, y, rs, popt):
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111)
    ax.plot(conc_bin_centers, nucleated_fractions, 'go-', label='Nucleation fraction')
    ax.plot(conc_bin_centers, y, 'r-', label='Logistic Function (a=1.00, b=%.2f, c=%.2f) [ $R^2 = %.3f$ ]' % (popt[0], popt[1], rs))
    ax.set_title('Nucleated Fractions across concentration slices ($R^2 = %.3f$)\n(%s)' % (rs, filename.split('.')[0]))
    ax.set_ylim(0, 1.1)
    ax.axhline(1.0, linestyle='--')
    ax.set_ylabel('Fraction Nucleated')
    ax.set_xlabel('Log10 Concentration')
    ax.legend(loc=8)

    savename = os.path.join(savepath, 'logistic-fits---%s.png' % filename.split('.')[0])
    fig.savefig(savename)


def generate_fine_grid_plots(savepath, config, data, filename):
    hist, _xedges, _yedges = np.histogram2d(data['concentration'], data['FRET'], bins=(config.fg_conc_bins, config.fg_fret_bins))
    hist[hist == 0] = None

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, aspect='equal')
    im = ax.imshow(np.rot90(hist), cmap='jet')

    ax.set_title(filename)
    ax.set_xlabel('Log10 Concentration')
    ax.set_ylabel('AmFRET/Concentration')
    ax.set_xticks(np.linspace(0, 300, 6))
    ax.set_xticklabels(['%.2f' % x for x in np.linspace(config.low_conc, config.high_conc, 6)])
    ax.set_yticks(np.linspace(0, 300, 5))
    ax.set_yticklabels(['%.2f' % y for y in np.linspace(config.high_fret, config.low_fret, 5)])
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='3%', pad=0.2)
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.set_ylabel('Counts', labelpad=30, rotation=270, fontsize=20)

    savename = os.path.join(savepath, '%s.png' % filename.split('.')[0])
    plt.savefig(savename)