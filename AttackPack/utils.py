import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.gridspec import GridSpec
from .database import AttackEntry
from .main import create_dataset


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
        - 'n': show noise only (requires modified_entry)
        - 'b': show noise in separate subplot alongside original/modified
        - 't:Title': set custom title
    """
    show_rr = 'r' in options
    separate_plots = 's' in options and modified_entry is not None
    separate_rr = 'd' in options and modified_entry is not None
    show_noise_only = 'n' in options and modified_entry is not None
    show_noise_beside = 'b' in options and modified_entry is not None
    title = options.split('t:')[1].strip() if 't:' in options else None

    ecg_data, rr_data, label = map(np.ndarray.flatten, entry)
    label_name = np.argmax(label)

    x_ecg, x_rr = np.arange(len(ecg_data)), np.arange(len(rr_data))

    # Calculate noise if modified entry is provided
    ecg_noise, rr_noise = None, None
    if modified_entry is not None:
        mod_ecg_data, mod_rr_data, mod_label = map(np.ndarray.flatten, modified_entry)
        ecg_noise = mod_ecg_data - ecg_data
        rr_noise = mod_rr_data - rr_data

    # Special case: show only noise
    if show_noise_only:
        rows = 2 if show_rr else 1
        fig = plt.figure(figsize=(12, 8 if rows == 2 else 6))
        gs = GridSpec(rows, 1, height_ratios=[3, 1] if rows == 2 else [1])

        # Plot ECG noise
        ax_ecg = fig.add_subplot(gs[0])
        ax_ecg.plot(x_ecg, ecg_noise, 'g-', linewidth=2, label='ECG Noise')
        ax_ecg.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax_ecg.set(title=title or 'Adversarial Noise',
                   xlabel='Sample', ylabel='Noise Amplitude')
        ax_ecg.grid(True, alpha=0.3)
        ax_ecg.legend()

        # Add noise statistics
        noise_stats = f'Max: {np.max(np.abs(ecg_noise)):.4f}, Std: {np.std(ecg_noise):.4f}, Mean: {np.mean(ecg_noise):.4f}'
        ax_ecg.text(0.02, 0.98, noise_stats, transform=ax_ecg.transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # Plot RR noise if requested
        if show_rr:
            ax_rr = fig.add_subplot(gs[1])
            bars = ax_rr.bar(x_rr, rr_noise, color='green', alpha=0.7, label='RR Noise')
            ax_rr.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            ax_rr.set(title='RR Interval Noise', xlabel='Interval Number', ylabel='Noise Amplitude')
            ax_rr.set_xticks(x_rr)
            ax_rr.grid(True, axis='y', alpha=0.3)
            ax_rr.legend()

            # Add value labels on bars
            for bar, value in zip(bars, rr_noise):
                height = bar.get_height()
                ax_rr.text(bar.get_x() + bar.get_width() / 2., height + np.sign(height) * 0.01 * max(abs(rr_noise)),
                           f'{value:.3f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)

        plt.tight_layout()
        plt.show()
        return fig

    # Determine subplot layout
    base_rows = 2 if show_rr or separate_plots else 1
    noise_rows = 2 if show_noise_beside and show_rr else 1 if show_noise_beside else 0
    additional_rows = 2 if separate_rr else 0

    rows = base_rows + noise_rows + additional_rows

    # Create height ratios
    height_ratios = []
    if separate_plots:
        height_ratios.extend([3, 3])  # Original and modified ECG
    else:
        height_ratios.append(3)  # Single ECG plot

    if show_rr and not separate_rr:
        height_ratios.append(1)  # Single RR plot
    elif separate_rr:
        height_ratios.extend([1, 1])  # Separate RR plots

    if show_noise_beside:
        height_ratios.append(2)  # ECG noise plot
        if show_rr:
            height_ratios.append(1)  # RR noise plot

    fig = plt.figure(figsize=(12, 2.5 * rows))
    gs = GridSpec(rows, 1, height_ratios=height_ratios)
    current_subplot = 0

    def plot_ecg(ax, data, color, label_text, title_text):
        ax.plot(x_ecg, data, color, label=label_text, linewidth=1.5)
        ax.set(title=title_text, xlabel='Sample', ylabel='Amplitude')
        ax.grid(True, alpha=0.3)
        ax.legend()

    def plot_rr(ax, data, color, label_text, title_text):
        bars = ax.bar(x_rr, data, color=color, alpha=0.7, label=label_text)
        ax.set(title=title_text, xlabel='Interval Number', ylabel='Value')
        ax.set_xticks(x_rr)
        ax.grid(True, axis='y', alpha=0.3)
        ax.legend()
        return bars

    def plot_noise(ax, noise_data, x_data, color, label_text, title_text, plot_type='line'):
        if plot_type == 'line':
            ax.plot(x_data, noise_data, color, linewidth=2, label=label_text)
            # Add statistics
            stats = f'Max: {np.max(np.abs(noise_data)):.4f}, Std: {np.std(noise_data):.4f}'
            ax.text(0.02, 0.98, stats, transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        else:  # bar
            bars = ax.bar(x_data, noise_data, color=color, alpha=0.7, label=label_text)
            # Add value labels
            for bar, value in zip(bars, noise_data):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height + np.sign(height) * 0.01 * max(abs(noise_data)),
                        f'{value:.3f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)
            ax.set_xticks(x_data)

        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax.set(title=title_text, xlabel='Sample' if plot_type == 'line' else 'Interval Number',
               ylabel='Noise Amplitude')
        ax.grid(True, alpha=0.3)
        ax.legend()

    # ECG Plots
    if separate_plots:
        plot_ecg(fig.add_subplot(gs[current_subplot]), ecg_data, 'b-', 'Original ECG',
                 f'Original ECG - Label: {label_name}')
        current_subplot += 1
        mod_label_name = np.argmax(mod_label)
        plot_ecg(fig.add_subplot(gs[current_subplot]), mod_ecg_data, 'r-', 'Modified ECG',
                 f'Modified ECG - Label: {mod_label_name}')
        current_subplot += 1
    else:
        ax = fig.add_subplot(gs[current_subplot])
        plot_ecg(ax, ecg_data, 'b-', 'Original ECG', title or f'ECG Data - Label: {label_name}')
        if modified_entry is not None:
            ax.plot(x_ecg, mod_ecg_data, 'r-', alpha=0.7, label='Modified ECG', linewidth=1.5)
            ax.legend()
        current_subplot += 1

    # RR Plots
    if show_rr:
        if separate_rr:
            plot_rr(fig.add_subplot(gs[current_subplot]), rr_data, 'blue', 'Original RR', 'Original RR Intervals')
            current_subplot += 1
            plot_rr(fig.add_subplot(gs[current_subplot]), mod_rr_data, 'red', 'Modified RR', 'Modified RR Intervals')
            current_subplot += 1
        else:
            ax = fig.add_subplot(gs[current_subplot])
            ax.bar(x_rr - 0.2, rr_data, width=0.4, color='blue', alpha=0.7, label='Original RR')
            if modified_entry is not None:
                ax.bar(x_rr + 0.2, mod_rr_data, width=0.4, color='red', alpha=0.7, label='Modified RR')
            ax.set(title='RR Intervals', xlabel='Interval Number', ylabel='Value')
            ax.set_xticks(x_rr)
            ax.grid(True, axis='y', alpha=0.3)
            ax.legend()
            current_subplot += 1

    # Noise Plots
    if show_noise_beside:
        # ECG Noise
        plot_noise(fig.add_subplot(gs[current_subplot]), ecg_noise, x_ecg, 'g-', 'ECG Noise', 'Adversarial Noise - ECG')
        current_subplot += 1

        # RR Noise
        if show_rr:
            plot_noise(fig.add_subplot(gs[current_subplot]), rr_noise, x_rr, 'green', 'RR Noise',
                       'Adversarial Noise - RR', 'bar')
            current_subplot += 1

    plt.tight_layout()
    plt.show()
    return fig


def show_entry(entry: AttackEntry):
    """
    Visualize a single attack entry.

    Parameters:
    -----------
    entry : AttackEntry
        The attack entry to visualize.
    """

    original_data = create_dataset(entry.data_file, torch.device("cpu"), dtype=torch.float32)
    original_data = original_data[entry.data_idx]

    # Create modified data based on attack extent
    if entry.extent == 'data':
        m_d = original_data[0] + entry.epsilon * entry.data_noise
        m_rr = original_data[1]
    elif entry.extent == 'rr':
        m_d = original_data[0]
        m_rr = original_data[1] + entry.epsilon * entry.rr_noise
    elif entry.extent == 'both':
        m_d = original_data[0] + entry.epsilon * entry.data_noise
        m_rr = original_data[1] + entry.epsilon * entry.rr_noise
    else:
        raise ValueError(f"Unknown extent: {entry.extent}")

    modified_data = (m_d, m_rr, original_data[2])

    return visualize_entry(
        tuple(t.numpy() for t in original_data),
        tuple(t.numpy() for t in modified_data),
        options=f"rb:t:{entry.model_name} - {entry.attack_type} - eps={entry.epsilon:.3f} - idx={entry.data_idx} - {entry.extent}"
    )


def show_noise_only(entry: AttackEntry):
    """
    Show only the adversarial noise for a given attack entry.

    Parameters:
    -----------
    entry : AttackEntry
        The attack entry to visualize noise for.
    """
    original_data = create_dataset(entry.data_file, torch.device("cpu"), dtype=torch.float32)
    original_data = original_data[entry.data_idx]

    # Create modified data based on attack extent
    if entry.extent == 'data':
        m_d = original_data[0] + entry.epsilon * entry.data_noise
        m_rr = original_data[1]
    elif entry.extent == 'rr':
        m_d = original_data[0]
        m_rr = original_data[1] + entry.epsilon * entry.rr_noise
    elif entry.extent == 'both':
        m_d = original_data[0] + entry.epsilon * entry.data_noise
        m_rr = original_data[1] + entry.epsilon * entry.rr_noise
    else:
        raise ValueError(f"Unknown extent: {entry.extent}")

    modified_data = (m_d, m_rr, original_data[2])

    return visualize_entry(
        tuple(t.numpy() for t in original_data),
        tuple(t.numpy() for t in modified_data),
        options=f"rn:t:NOISE - {entry.model_name} - {entry.attack_type} - eps={entry.epsilon:.3f}"
    )