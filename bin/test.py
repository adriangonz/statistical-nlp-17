"""
Command to test a stored model against the training set.
"""
import os
import torch

from argparse import ArgumentParser

from sklearn.metrics import accuracy_score

from torch.utils.data import DataLoader

from src.vocab import VanillaVocab
from src.matching_network import MatchingNetwork
from src.evaluation import (predict, save_predictions, generate_episode_data)
from src.datasets import EpisodesSampler, EpisodesDataset
from src.utils import extract_model_parameters, get_model_name

BATCH_SIZE = 64

parser = ArgumentParser()
parser.add_argument(
    "-v",
    "--vocab",
    action="store",
    dest="vocab",
    type=str,
    help="Path to the vocab JSON file")
parser.add_argument(
    "-m",
    "--model",
    action="store",
    dest="model",
    type=str,
    help="Path to the stored model snapshot")
parser.add_argument(
    "-p",
    "--store-predictions",
    action="store_true",
    dest="predictions",
    default=False,
    help="If enabled, store predictions")
parser.add_argument(
    "-e",
    "--generate-episode-data",
    action="store_true",
    dest="episode",
    default=False,
    help="If enabled, generate data for a single episode")
parser.add_argument("test_set", help="Path to the test CSV file")


def _load_model(model_path):
    model_file_name = os.path.basename(args.model)
    distance, embeddings, N, k = extract_model_parameters(model_file_name)
    model_name = get_model_name(distance, embeddings, N, k)
    model = MatchingNetwork(model_name, distance_metric=distance)
    model_state_dict = torch.load(model_path)
    model.load_state_dict(model_state_dict)

    return model, embeddings, N, k


def main(args):
    print("Loading model...")
    model, _, N, k = _load_model(args.model)

    print("Loading dataset...")
    vocab = VanillaVocab(args.vocab)
    X_test, y_test = vocab.to_tensors(args.test_set)
    test_set = EpisodesDataset(X_test, y_test, k=k)
    sampler = EpisodesSampler(test_set, N=N, episodes_multiplier=30)
    test_loader = DataLoader(test_set, sampler=sampler, batch_size=BATCH_SIZE)

    print("Evaluating model...")
    labels, predictions = predict(model, test_loader)
    np_labels = labels.numpy()
    np_predictions = predictions.numpy()

    print("Storing results...")
    if args.predictions:
        print("Storing predictions...")
        save_predictions(model, np_labels, np_predictions)

    # Get a single batch
    if args.episode:
        print("Generating data for a single episode...")
        correct = generate_episode_data(model, test_loader, vocab)
        # Accuracy is not very high, so we want to make sure the label
        # right
        if not correct:
            print("[WARNING] Predicted example was incorrect!")

    # Compute accuracy
    accuracy = accuracy_score(np_labels, np_predictions)
    print("==========================")
    print(f"Accuracy = {accuracy:.3f}")
    print("==========================")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
