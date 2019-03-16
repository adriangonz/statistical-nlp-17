"""
Command to test a stored model against the training set.
"""
import os
import torch

from argparse import ArgumentParser

from sklearn.metrics import accuracy_score

from torch.utils.data import DataLoader

from src.matching_network import MatchingNetwork
from src.evaluation import predict
from src.data import read_vocab, read_data_set
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
    vocab = read_vocab(args.vocab)
    X_test, y_test = read_data_set(args.test_set, vocab)
    test_set = EpisodesDataset(X_test, y_test, k=k)
    sampler = EpisodesSampler(test_set, N=N, episodes_multiplier=20)
    test_loader = DataLoader(test_set, sampler=sampler, batch_size=BATCH_SIZE)

    print("Evaluating model...")
    labels, predictions = predict(model, test_loader)
    accuracy = accuracy_score(labels, predictions)
    print(f"ACCURACY = {accuracy:.3f}")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
