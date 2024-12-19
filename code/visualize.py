import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_metric(
    metrics,
    metric_name,
    levels=None,
    title=None,
    xlabel="Time",
    ylabel="Metric Value",
    figsize=(10, 6),
    max_time=None,
    save_path=None
):
    """
    Plots a single specified metric with high-quality visuals suitable for research papers.

    Parameters:
        metrics (dict): A dictionary of metrics with keys like "timestamps", "naive_biases", "population_biases", etc.
        metric_name (str): The name of the metric to plot (e.g., "naive_biases").
        levels (list, optional): List of levels to plot. Defaults to None (entire company only).
        title (str, optional): Title for the plot.
        xlabel (str, optional): Label for the x-axis. Defaults to "Time".
        ylabel (str, optional): Label for the y-axis. Defaults to "Metric Value".
        figsize (tuple, optional): Size of the figure. Defaults to (10, 6).
        max_time (float, optional): Maximum time to set the x-axis range. Defaults to max(timestamps).
        save_path (str, optional): File path to save the plot. If None, the plot is not saved.

    Returns:
        None: Displays the plot and optionally saves it.
    """
    plt.style.use("default")  # Use the default Matplotlib color scheme
    timestamps = metrics["timestamps"]
    max_time = max_time or max(timestamps)
    levels = levels or ["company"]  # Default to entire company if no levels specified

    fig, ax = plt.subplots(figsize=figsize)

    for idx, level in enumerate(levels):
        values = metrics[metric_name][level]
        label = f"{metric_name.capitalize()} ({'Overall' if level == 'company' else f'Level {level}'})"
        ax.plot(timestamps, values, label=label, linewidth=2, alpha=0.9)

    # Set x-axis range and ticks
    ax.set_xlim(0, max_time)
    ax.set_xticks(np.linspace(0, max_time, num=6))  # Adjust tick intervals
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=10)
    ax.grid(True, which="both", linestyle="--", linewidth=0.7, color="gray")  # Darker grid

    # Title and labels
    ax.set_title(title or f"{metric_name.capitalize()} Trends Over Time", fontsize=14)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(title="Levels", fontsize=10, title_fontsize=12, loc="upper right", frameon=True)
    plt.tight_layout()

    # Save plot to file for research paper usage (optional)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    plt.show()

def plot_identity_percentages(
    percentages,
    timestamps,
    identities,
    level=None,
    max_time=None,
    population_percentages=None,
    title=None,
    save_path=None
):
    """
    Plot identity percentages over time with shaded regions representing each identity.

    Parameters:
        percentages (dict): A dictionary of identity percentages over time.
        timestamps (list): List of timestamps.
        identities (list): List of identity groups (e.g., ["F", "M"]).
        level (int, optional): Level to plot percentages for. Defaults to None (entire company).
        max_time (float, optional): Maximum time to set the x-axis range. Defaults to max(timestamps).
        population_percentages (dict, optional): A dictionary with the expected population percentages for each identity.
        title (str, optional): Title for the plot.
        save_path (str, optional): File path to save the plot. If None, the plot is not saved.

    Returns:
        None: Displays the plot and optionally saves it.
    """
    # Extract data for the specified level or company-wide
    data = percentages["company"] if level is None else percentages[level]
    max_time = max_time or max(timestamps)

    fig, ax = plt.subplots(figsize=(10, 6))
    cumulative = np.zeros(len(timestamps))
    cumulative_population_percentage = 0  # Tracks cumulative population percentage

    for i, identity in enumerate(identities):
        identity_percentages = np.array(data[identity])

        ax.step(timestamps, cumulative + identity_percentages, where="post")
        ax.fill_between(
            timestamps,
            cumulative,
            cumulative + identity_percentages,
            label=identity,
            alpha=0.7
        )
        # Add horizontal line for cumulative population percentage if it's < 100%
        if population_percentages and identity in population_percentages:
            cumulative_population_percentage += population_percentages[identity]
            if cumulative_population_percentage < 1:
                # Add shadow by plotting a thicker, lighter line behind the main line
                # ax.axhline(
                #     y=cumulative_population_percentage,
                #     color="black",
                #     linestyle="-",
                #     linewidth=3,
                #     alpha=0.8
                # )
                # Main line with black border for visibility
                ax.axhline(
                    y=cumulative_population_percentage,
                    color="red",
                    linestyle="-",
                    linewidth=1.5
                )
        cumulative += identity_percentages

    # Set x-axis range and ticks
    ax.set_xlim(0, max_time)
    ax.set_xticks(np.linspace(0, max_time, num=6))  # Adjust tick intervals
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=10)
    ax.grid(True, which="both", linestyle="--", linewidth=0.7, color="gray")  # Darker grid

    # Title and labels
    ax.set_title(title or f"Identity Percentages Over Time ({'Overall' if level is None else f'Level {level}'})", fontsize=14)
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Percentage", fontsize=12)
    ax.set_ylim(0, 1)
    ax.legend(title="Identities", fontsize=10, title_fontsize=12, loc="upper left", frameon=True)
    plt.tight_layout()

    # Save plot to file for research paper usage (optional)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    plt.show()

