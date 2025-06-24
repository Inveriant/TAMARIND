import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import os.path as pathutils
from matplotlib.lines import Line2D
import os

from algorithms.problem_variants import eh_rsa, eh_rsa_orig, eh_rsa_all_surviving_estimates
from algorithms.cost_estimator import rank_estimate
from utils.helpers import significant_bits


def plot(key_size: int = 1024):
    # Try to load cached data first
    cache_file = f'plot_cache_{key_size}.npy'
    if pathutils.exists(cache_file):
        print(f"Loading cached data from {cache_file}")
        cached_data = np.load(cache_file, allow_pickle=True).item()
        plot_from_cache(cached_data, key_size)
        return

    # Choose bit sizes to plot if no cache exists
    min_key_size = key_size
    if key_size == 1024:
        max_steps = 64
    else:
        max_steps = 8
    bits = [min_key_size * s for s in range(1, max_steps + 1)]
    bits = [e for e in bits if significant_bits(e) <= 3]
    max_y = min_key_size * max_steps

    datasets = [
        ('C0', 'Our work - 0.1% gate error rate', eh_rsa, 1e-3, 'o'),
        ('C1', 'Our work - 0.01% gate error rate', eh_rsa, 1e-4, '*'),
        ('C2', 'Ekerå-Håstad - 0.1% gate error rate', eh_rsa_orig, 1e-3, 's'),
        ('C3', 'Ekerå-Håstad - 0.01% gate error rate', eh_rsa_orig, 1e-4, 'd'),
    ]

    # Adjust figure size for small key sizes
    if key_size <= 256:
        plt.subplots(figsize=(20, 11))  # Increased height for legend
    else:
        plt.subplots(figsize=(16, 9))
    plt.rcParams.update({'font.size': 14})

    # Dictionary to store results for caching
    cache_data = {
        'bits': bits,
        'max_y': max_y,
        'results': []
    }

    def process_dataset(color, name, func, gate_error_rate, marker):
        valid_ns = []
        hours = []
        megaqubits = []

        for n in bits:
            cost = func(n, gate_error_rate)
            if cost is not None:
                expected_hours = cost.total_hours / (1 - cost.total_error)
                hours.append(expected_hours)
                megaqubits.append(cost.total_megaqubits)
                valid_ns.append(n)

        return color, name, marker, valid_ns, hours, megaqubits

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_dataset, color, name, func, gate_error_rate, marker)
                   for color, name, func, gate_error_rate, marker in datasets]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Datasets"):
            color, name, marker, valid_ns, hours, megaqubits = future.result()
            cache_data['results'].append({
                'color': color,
                'name': name,
                'marker': marker,
                'valid_ns': valid_ns,
                'hours': hours,
                'megaqubits': megaqubits
            })
            plt.plot(valid_ns, hours, color=color, label=name + ', hours', marker=marker)
            plt.plot(valid_ns, megaqubits, color=color, label=name + ', megaqubits', linestyle='--', marker=marker)

    # Save the computed data
    np.save(cache_file, cache_data)
    print(f"Cached data saved to {cache_file}")

    finish_plot(bits, max_y, key_size)


def plot_from_cache(cache_data, key_size):
    plt.subplots(figsize=(16, 9))
    plt.rcParams.update({'font.size': 14})

    bits = cache_data['bits']
    max_y = cache_data['max_y']

    for result in cache_data['results']:
        plt.plot(result['valid_ns'], result['hours'], 
                color=result['color'], 
                label=result['name'] + ', hours', 
                marker=result['marker'])
        plt.plot(result['valid_ns'], result['megaqubits'],
                color=result['color'],
                label=result['name'] + ', megaqubits',
                linestyle='--',
                marker=result['marker'])

    finish_plot(bits, max_y, key_size)


def finish_plot(bits, max_y, key_size):
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(key_size, max_y)
    plt.xticks(bits, [str(e) for e in bits], rotation=90, fontsize=12)
    
    if key_size <= 256:
        yticks = [(5 if e else 1) * 10**k
                for k in range(3)  # Only go up to 10^2 = 100
                for e in range(2)][:-1]
        plt.ylim(top=500)
    else:
        yticks = [(5 if e else 1) * 10**k
                for k in range(6)
                for e in range(2)][:-1]
                
    plt.yticks(yticks, [str(e) for e in yticks], fontsize=12)
    plt.minorticks_off()
    plt.grid(True)
    plt.xlabel('Modulus length n (bits)', fontsize=16, fontweight='bold')
    plt.ylabel('Expected time (hours) and physical qubit count (megaqubits)', fontsize=16, fontweight='bold')

    # Place legend inside the axes in the upper-left corner
    plt.legend(loc='upper left', shadow=False, fontsize=14)
    plt.gcf().subplots_adjust(bottom=0.16)  # Adjust bottom to make room for ticks
    
    plt.tight_layout()
    path = f'assets/{str(key_size)}-rsa-dlps-extras.pdf'
    plt.savefig(path, bbox_inches='tight')


def plot_surviving_estimates(key_sizes: list[int], gate_error_rate: float):
    """Creates a scatter plot of all surviving estimates for multiple key sizes."""
    # Set up the plot style
    plt.subplots(figsize=(16, 9))
    plt.rcParams.update({'font.size': 14})
    
    # Use consistent colors from the rest of the code
    color_map = {
        1024: 'C0',  # Blue
        2048: 'C1',  # Orange
        3072: 'C2',  # Green
        4096: 'C3',  # Red
        8192: 'C4',  # Purple
        12288: 'C5', # Brown
        16384: 'C6'  # Pink
    }
    
    # Store best estimates for annotation
    best_estimates = {}
    
    # Process each key size with tqdm progress bar
    min_time = float('inf')
    max_time = 0
    min_qubits = float('inf')
    max_qubits = 0
    
    for idx, n in tqdm(enumerate(key_sizes), total=len(key_sizes), desc="Processing key sizes"):
        surviving_estimates = eh_rsa_all_surviving_estimates(n, gate_error_rate)
        
        if not surviving_estimates:
            print(f"No surviving estimates for n={n}")
            continue
            
        expected_times = [est.total_hours / (1 - est.total_error) for est in surviving_estimates]
        qubit_counts = [est.total_megaqubits for est in surviving_estimates]
        
        # Plot points with low alpha
        plt.scatter(expected_times, qubit_counts, 
                   alpha=0.2,  # Low transparency for actual points
                   c=color_map[n],
                   s=100,
                   rasterized=False,
                   label='_nolegend_')  # Don't include in legend
        
        best = min(surviving_estimates, key=rank_estimate)
        best_estimates[n] = best
        
        best_time = best.total_hours / (1 - best.total_error)
        best_qubits = best.total_megaqubits
        plt.scatter([best_time], [best_qubits], 
                   color=color_map[n],
                   marker='$\u26BF$',  # Diamond marker for minimum estimates
                   s=200,
                   zorder=5,
                   edgecolor='black',
                   linewidth=1,
                   alpha=1.0,
                   label='_nolegend_')  # Don't add duplicate legend entries
        
        min_time = min(min_time, min(expected_times))
        max_time = max(max_time, max(expected_times))
        min_qubits = min(min_qubits, min(qubit_counts))
        max_qubits = max(max_qubits, max(qubit_counts))
    
    # Set up axes
    plt.xscale('log')
    plt.yscale('log')
    
    # Generate tick locations
    def generate_ticks(min_val, max_val):
        ticks = []
        start_exp = int(np.floor(np.log10(min_val)))
        end_exp = int(np.ceil(np.log10(max_val)))
        
        for exp in range(start_exp, end_exp + 1):
            for base in [1, 2, 5]:
                tick = base * 10**exp
                if min_val <= tick <= max_val:
                    ticks.append(tick)
        return ticks
    
    x_ticks = generate_ticks(min_time, max_time)
    y_ticks = generate_ticks(min_qubits, max_qubits)
    
    plt.xticks(x_ticks, [f'{t:.1f}' for t in x_ticks], rotation=45, fontsize=12)
    plt.yticks(y_ticks, [f'{t:.1f}' for t in y_ticks], fontsize=12)
    
    plt.minorticks_off()  # Match other plots
    plt.grid(True)  # Simple grid like other plots
    
    plt.xlabel('Expected time (hours)', fontsize=16, fontweight='bold')
    plt.ylabel('Physical qubit count (millions)', fontsize=16, fontweight='bold')
    
    # Create custom legend handles
    legend_handles = []
    for n, color in color_map.items():
        if n in best_estimates:
            # Create a solid marker for the legend
            handle = Line2D([0], [0], marker='o', color='none', 
                            markerfacecolor=color, alpha=1.0, 
                            markersize=10, label=f'n={n} bits')
            legend_handles.append(handle)

    # Add the minimum estimate marker
    min_handle = Line2D([0], [0], marker='$\u26BF$', color='gray', 
                        label='Minimum estimate', markersize=10, 
                        markerfacecolor='gray', linestyle='None')
    legend_handles.append(min_handle)
    
    # Add the legend to the plot
    plt.legend(handles=legend_handles, 
               title='Key Sizes', 
               title_fontsize=14, 
               fontsize=14, 
               loc='upper left', 
               shadow=False)
    
    plt.gcf().subplots_adjust(bottom=0.16)
    plt.tight_layout()
    
    # Save plot
    # base_path = pathutils.dirname(pathutils.dirname(pathutils.realpath(__file__)))
    # Ensure the directory exists
    directory = pathutils.dirname(pathutils.realpath(__file__))
    assets_dir = pathutils.normpath(directory + '/assets')
    if not pathutils.exists(assets_dir):
        os.makedirs(assets_dir)

    path = pathutils.normpath(assets_dir + f'/surviving-estimates-{gate_error_rate:.0e}.png')
    plt.savefig(path, bbox_inches='tight', dpi=300, transparent=True)
    plt.savefig(path, bbox_inches='tight')
    print(f"\nPlot saved to: {path}")


def update_cached_labels(key_size: int = 1024):
    cache_file = f'plot_cache_{key_size}.npy'
    if pathutils.exists(cache_file):
        cached_data = np.load(cache_file, allow_pickle=True).item()
        
        # Define the mapping from old to new labels
        new_names = {
            'RSA via Optimized Windowing - 0.1% gate error rate': 'Our work - 0.1% gate error rate',
            'RSA via Optimized Windowing - 0.01% gate error rate': 'Our work - 0.01% gate error rate',
            'RSA via Ekerå-Håstad - 0.1% gate error rate': 'Ekerå-Håstad - 0.1% gate error rate',
            'RSA via Ekerå-Håstad - 0.01% gate error rate': 'Ekerå-Håstad - 0.01% gate error rate'
        }
        
        # Update the names in the cached data
        for result in cached_data['results']:
            if result['name'] in new_names:
                result['name'] = new_names[result['name']]
        
        # Save the updated data back to the cache file
        np.save(cache_file, cached_data)
        print(f"Updated labels in {cache_file}")
