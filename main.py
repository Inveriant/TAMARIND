import matplotlib.pyplot as plt

from output.tabulation import tabulate
from visualization.plotting import plot


def main():
    print("Hello from tamarind!")

    # Example usage
    
    # Uncomment to generate tables
    # tabulate()
    
    # Uncomment to update cached labels
    # for key_size in [256, 1024]:  # add or remove sizes as needed
    #     update_cached_labels(key_size)
    
    # Uncomment to generate plots
    plot(key_size=256)
    # plot()    
    
    # Example: Plot surviving estimates for multiple key sizes
    # key_sizes = [1024, 2048, 3072, 4096]
    # plot_surviving_estimates(key_sizes, gate_error_rate=1e-3)
    # plt.show()


if __name__ == "__main__":
    main()
