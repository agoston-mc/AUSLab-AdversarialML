import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def visualize_entry(entry, modified_entry=None, options=""):
    """
    Visualize ECG data with optional comparison to modified data.

    Parameters:
    -----------
    entry : tuple (ecg_data, rr_data, label)
        - ecg_data: ECG signal array of shape (1, 300)
        - rr_data: RR intervals array of shape (1, 4)
        - label: one-hot encoded array of size 5 for classification

    modified_entry : tuple, optional
        Same structure as entry for comparison

    options : str, optional
        - 'r': show RR intervals
        - 's': separate plots for original and modified ECG
        - 'd': separate plots for RR intervals
        - 't:Title': set custom title
    """
    show_rr = 'r' in options
    separate_plots = 's' in options and modified_entry is not None
    separate_rr = 'd' in options and modified_entry is not None
    title = options.split('t:')[1].strip() if 't:' in options else None

    ecg_data, rr_data, label = map(np.ndarray.flatten, entry)
    # todo dunno, and also at the modified
    # label_names = ['Don"t', 'know']
    label_name = np.argmax(label)

    x_ecg, x_rr = np.arange(len(ecg_data)), np.arange(len(rr_data))

    # Setup figure
    rows = 4 if show_rr and separate_plots and separate_rr else 2 if show_rr or separate_plots else 1
    height_ratios = [3]*min(rows,2) + ([1]* (rows - 2))
    fig = plt.figure(figsize=(12, 10 if rows > 2 else 8 if rows == 2 else 6))
    gs = GridSpec(rows, 1, height_ratios=height_ratios)

    def plot_ecg(ax, data, color, label_text, title_text):
        ax.plot(x_ecg, data, color, label=label_text)
        ax.set(title=title_text, xlabel='Sample', ylabel='Amplitude')
        ax.grid(True)
        ax.legend()

    def plot_rr(ax, data, color, label_text, title_text):
        ax.bar(x_rr, data, color=color, alpha=0.7, label=label_text)
        ax.set(title=title_text, xlabel='Interval Number', ylabel='Value')
        ax.set_xticks(x_rr)
        ax.grid(True, axis='y')
        ax.legend()

    # ECG Plots
    if separate_plots:
        plot_ecg(fig.add_subplot(gs[0]), ecg_data, 'b-', 'Original ECG', f'Original ECG - Label: {label_name}')
        mod_ecg_data, mod_rr_data, mod_label = map(np.ndarray.flatten, modified_entry)
        mod_label_name = np.argmax(mod_label)
        plot_ecg(fig.add_subplot(gs[1]), mod_ecg_data, 'r-', 'Modified ECG', f'Modified ECG - Label: {mod_label_name}')
    else:
        ax = fig.add_subplot(gs[0])
        plot_ecg(ax, ecg_data, 'b-', 'Original ECG', title or f'ECG Data - Label: {label_name}')
        if modified_entry is not None:
            mod_ecg_data = modified_entry[0].flatten()
            ax.plot(x_ecg, mod_ecg_data, 'r-', alpha=0.7, label='Modified ECG')
            ax.legend()

    # RR Plots
    if show_rr:
        if separate_rr:
            plot_rr(fig.add_subplot(gs[2]), rr_data, 'blue', 'Original RR', 'Original RR Intervals')
            mod_rr_data = modified_entry[1].flatten()
            plot_rr(fig.add_subplot(gs[3]), mod_rr_data, 'red', 'Modified RR', 'Modified RR Intervals')
        else:
            ax = fig.add_subplot(gs[-1])
            ax.bar(x_rr - 0.2, rr_data, width=0.4, color='blue', alpha=0.7, label='Original RR')
            if modified_entry is not None:
                mod_rr_data = modified_entry[1].flatten()
                ax.bar(x_rr + 0.2, mod_rr_data, width=0.4, color='red', alpha=0.7, label='Modified RR')
            ax.set(title='RR Intervals', xlabel='Interval Number', ylabel='Value')
            ax.set_xticks(x_rr)
            ax.grid(True, axis='y')
            ax.legend()

    plt.tight_layout()
    plt.show()
    return fig


