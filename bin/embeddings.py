import os
import numpy as np

from argparse import ArgumentParser

from src.figures import plot_embeddings, save_episode_text
from src.utils import extract_model_parameters, get_model_name

parser = ArgumentParser()
parser.add_argument("embeddings", help="Path to the stored model's embeddings")


def main(args):
    print("Loading data...")
    model_file_name = os.path.basename(args.embeddings)
    d, e, N, k = extract_model_parameters(model_file_name)
    model_name = get_model_name(d, e, N, k)

    results = np.load(args.embeddings)
    support_embeddings = results['support_embeddings']
    target_embeddings = results['target_embeddings']
    support_set = results['support_set']
    targets = results['targets']
    labels = results['labels']
    target_labels = results['target_labels']

    print("Generating and saving the embeddings plot...")
    plot_embeddings(model_name, support_embeddings, target_embeddings, labels)
    save_episode_text(
        model_name,
        support_set,
        targets,
        labels,
        target_labels,
        suffix='embeddings')


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
