import os
import csv

from matplotlib import pyplot as plt

FIGURES_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "figures")


def save_episode_text(model_name, support_set, targets, labels, target_labels,
                      suffix):
    """
    Saves the episode text to be able to refer to it on the
    report.

    Parameters
    ---
    model_name : str
        Model name used for filename.
    support_set : np.array[N x k]
        Sentences of the support set.
    targets : np.array[T]
        Sentences of the target set.
    labels : np.array[N]
        Labels of the entire episode.
    target_labels : np.array[T]
        Labels of the target set.
    suffix : str
        Suffix to add to the filename. Useful to differentiate between text
        saved for different figures.
    """
    N, _ = support_set.shape
    T, = targets.shape

    file_name = f"{model_name}_{suffix}_text.csv"
    file_path = os.path.join(FIGURES_PATH, file_name)
    with open(file_path, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(["label", "sentence"])

        # First store support set
        for label, examples in zip(labels, support_set):
            for example in examples:
                writer.writerow([label, example])

        # Then targets
        for label, target in zip(target_labels, targets):
            writer.writerow([label, target])


def plot_attention_map(model_name, attention, labels):
    """
    Plots the attention map for a single episode and target.

    Parameters
    ---
    model_name : str
        Model name that generated this plot.
    attention : np.array[T x N x k]
        Attention map over examples.
    labels : np.array[N]
        Labels for the episode.
    """
    target_attention = attention[0]

    fig, axis = plt.subplots()
    heatmap = axis.pcolor(target_attention, cmap=plt.cm.Blues)

    # had to do it manually
    axis.set_yticks([0.5, 1.5, 2.5, 3.5, 4.5], minor=False)
    axis.set_xticks([0.5, 1.5, 2.5], minor=False)

    # same here
    column_labels = [1, 2, 3]
    axis.invert_yaxis()
    plt.xlabel('Example (k)')
    plt.ylabel('Label (N)')
    axis.set_yticklabels(labels, minor=False)
    axis.set_xticklabels(column_labels, minor=False)
    plt.title('Attention map for examples in Support Set \n', fontsize=16)
    fig.set_size_inches(11.03, 7.5)
    plt.colorbar(heatmap)

    file_name = (f"{model_name}_heatmap.png")
    file_path = os.path.join(FIGURES_PATH, file_name)
    plt.savefig(file_path, dpi=300)
