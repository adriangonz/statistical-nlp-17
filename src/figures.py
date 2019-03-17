import os
import csv
import seaborn as sns
import numpy as np

from sklearn.manifold import MDS

from matplotlib import pyplot as plt

FIGURES_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "figures")

DPI = 300

sns.set()


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


def plot_embeddings(model_name, support_embeddings, target_embeddings, labels):
    """
    Plots the embeddings in 2D with t-SNE.

    Parameters
    ---
    model_name : str
        Model name that generated this plot.
    support_embeddings : np.array[N x k x encoding_size]
        Support set embeddings.
    target_embeddings : np.array[T x encoding_size]
        Target set embeddings.
    labels : np.array[N]
        Label texts.
    """
    N, k, encoding_size = support_embeddings.shape
    T, _ = target_embeddings.shape
    mds = MDS(n_components=2)

    # Combine embeddings into a single matrix
    flat_support_embeddings = support_embeddings.reshape(-1, encoding_size)
    all_embeddings = np.concatenate(
        [flat_support_embeddings, target_embeddings], axis=0)

    # Compute 2D projections
    all_points = mds.fit_transform(all_embeddings)
    flat_support_points = all_points[:N * k]
    support_points = flat_support_points.reshape(N, k, -1)
    target_points = all_points[-T:]

    # Build plot
    fig, ax = plt.subplots()

    # Plot support set points
    for label, examples_points in zip(labels, support_points):
        x = examples_points[:, 0]
        y = examples_points[:, 1]
        ax.scatter(x, y, label=label)

    # Plot target set point
    x = target_points[:, 0]
    y = target_points[:, 1]
    ax.scatter(x, y, s=2**8, marker='X')

    ax.legend()
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    file_name = (f"{model_name}_embeddings.png")
    file_path = os.path.join(FIGURES_PATH, file_name)
    plt.savefig(file_path, dpi=DPI)


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
    attention_map = axis.pcolor(target_attention, cmap=plt.cm.Blues)

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
    fig.set_size_inches(11.03, 7.5)
    plt.colorbar(attention_map)

    file_name = (f"{model_name}_attention.png")
    file_path = os.path.join(FIGURES_PATH, file_name)
    plt.savefig(file_path, dpi=DPI)
