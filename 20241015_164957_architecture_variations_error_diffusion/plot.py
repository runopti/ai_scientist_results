import json
import os
import os.path as osp
import pickle

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

# LOAD FINAL RESULTS:
datasets = ["gaussian", "xor", "circle", "spiral"]
act_fun = ["sigmoid"]
folders = os.listdir("./")
final_results = {}
train_info = {}


def smooth(x, window_len=10, window="hanning"):
    s = np.r_[x[window_len - 1 : 0 : -1], x, x[-2 : -window_len - 1 : -1]]
    if window == "flat":  # moving average
        w = np.ones(window_len, "d")
    else:
        w = getattr(np, window)(window_len)
    y = np.convolve(w / w.sum(), s, mode="valid")
    return y


for folder in folders:
    if folder.startswith("run") and osp.isdir(folder):
        with open(osp.join(folder, "final_info.json"), "r") as f:
            final_results[folder] = json.load(f)
        all_results = pickle.load(
            open(osp.join(folder, "all_results.pkl"), "rb")
        )
        train_info[folder] = all_results

# CREATE LEGEND -- PLEASE FILL IN YOUR RUN NAMES HERE
# Keep the names short, as these will be in the legend.
labels = {
    "run_0": "Single hidden layer with 8 neurons",
    "run_1": "Two hidden layers with 8 neurons each",
    "run_2": "Single hidden layer with 16 neurons",
    "run_3": "Single hidden layer with 32 neurons",
    "run_4": "Two hidden layers with 16 neurons each",
}

# Use the run key as the default label if not specified
runs = list(final_results.keys())
for run in runs:
    if run not in labels:
        labels[run] = run


# CREATE PLOTS


# Create a programmatic color palette
def generate_color_palette(n):
    cmap = plt.get_cmap(
        "tab20"
    )  # You can change 'tab20' to other colormaps like 'Set1', 'Set2', 'Set3', etc.
    return [mcolors.rgb2hex(cmap(i)) for i in np.linspace(0, 1, n)]


# Get the list of runs and generate the color palette
runs = list(final_results.keys())
colors = generate_color_palette(len(runs))

# Plot 1: Line plot of training loss for each dataset across the runs with labels
fig, axs = plt.subplots(2, 2, figsize=(14, 8), sharex=True)

for j, dataset in enumerate(datasets):
    row = j // 2
    col = j % 2
    for i, run in enumerate(runs):
        print(train_info[run]["gaussian_sigmoid"].keys())
        mean = train_info[run][dataset + "_" + act_fun[0]]["train_accuracies"]
        mean = smooth(mean, window_len=25)
        axs[row, col].plot(mean, label=labels[run], color=colors[i])
        axs[row, col].set_title(f"{dataset.capitalize()}")
        axs[row, col].legend()
        axs[row, col].set_xlabel("Training Step")
        axs[row, col].set_ylabel("Train Accuracy")

plt.tight_layout()
plt.savefig("train_acc.png")
# plt.show()
plt.close()

# Plot 2: Visualize generated samples
# If there is more than 1 run, these are added as extra rows.
num_runs = len(runs)
fig, axs = plt.subplots(num_runs, 4, figsize=(14, 3 * num_runs))

for i, run in enumerate(runs):
    for j, dataset in enumerate(datasets):
        plot_info = train_info[run][dataset + "_" + act_fun[0]]["plot_info"]
        if num_runs == 1:
            axs[j].contourf(
                plot_info["x1"],
                plot_info["x2"],
                plot_info["pred"],
                alpha=0.3,
                cmap="bwr",  # colors[i],
            )
            axs[j].scatter(
                plot_info["x"][:, 0],
                plot_info["x"][:, 1],
                c=plot_info["y"],
                alpha=0.8,
                cmap="bwr",
            )
            axs[j].set_title(
                f"{dataset.capitalize()}, Test Acc: {plot_info['test_acc']:.2f}"
            )
        else:
            axs[i, j].contour(
                plot_info["x1"],
                plot_info["x2"],
                plot_info["pred"],
                alpha=0.3,
                cmap="bwr",  # colors[i],
            )
            axs[i, j].scatter(
                plot_info["x"][:, 0],
                plot_info["x"][:, 1],
                c=plot_info["y"],
                cmap="bwr",
                alpha=0.8,
            )
            axs[i, j].set_title(
                f"{dataset.capitalize()}, Test Acc: {plot_info['test_acc']:.2f}"
            )
    if num_runs == 1:
        axs[0].set_ylabel(labels[run])
    else:
        axs[i, 0].set_ylabel(labels[run])

plt.tight_layout()
plt.savefig("generated_images.png")
# plt.show()
plt.close()
