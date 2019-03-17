"""
Command to generate an attention map over a target
of the episode.
"""
import numpy as np
import os

from argparse import ArgumentParser

from src.utils import get_model_name, extract_model_parameters
from src.figures import plot_attention_map, save_episode_text

parser = ArgumentParser()
parser.add_argument("attention", help="Path to the stored model's attentions")


def main(args):
    print("Loading data...")
    model_file_name = os.path.basename(args.attention)
    d, e, N, k = extract_model_parameters(model_file_name)
    model_name = get_model_name(d, e, N, k)

    results = np.load(args.attention)
    attention = results['attention']
    support_set = results['support_set']
    targets = results['targets']
    labels = results['labels']
    target_labels = results['target_labels']

    print('Generating and saving the plot...')
    plot_attention_map(model_name, attention, labels)
    save_episode_text(
        model_name,
        support_set,
        targets,
        labels,
        target_labels,
        suffix='attention')


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
