"""
Command to train the model.
"""
from argparse import ArgumentParser

from torch.utils.data import DataLoader

from src.data import read_vocab, read_data_set
from src.datasets import EpisodesSampler, EpisodesDataset
from src.matching_network import MatchingNetwork
from src.training import train
from src.utils import train_test_split_tensors

BATCH_SIZE = 32

parser = ArgumentParser()
parser.add_argument(
    "-N",
    "--labels-per-episode",
    action="store",
    dest="N",
    type=int,
    help="Number of labels per episode")
parser.add_argument(
    "-k",
    "--examples-per-episode",
    action="store",
    dest="k",
    type=int,
    help="Examples per label")
parser.add_argument("vocab", help="Path to the vocab JSON file")
parser.add_argument("training_set", help="Path to the training CSV file")


def _get_loader(data_set, N):
    sampler = EpisodesSampler(data_set, N=N)
    loader = DataLoader(data_set, sampler=sampler, batch_size=BATCH_SIZE)
    return loader


def main(args):
    print("Loading dataset...")
    vocab = read_vocab(args.vocab)
    X_train, y_train = read_data_set(args.training_set, vocab)

    # Split training further into train and valid
    X_train, X_valid, y_train, y_valid = train_test_split_tensors(
        X_train, y_train, test_size=0.3)
    train_set = EpisodesDataset(X_train, y_train, k=args.k)
    valid_set = EpisodesDataset(X_valid, y_valid, k=args.k)

    print("Initialising model...")
    model = MatchingNetwork(fce=True, processing_steps=3)

    print("Starting to train...")
    train_loader = _get_loader(train_set, args.N)
    valid_loader = _get_loader(valid_set, args.N)
    train(
        model,
        learning_rate=1e-3,
        train_loader=train_loader,
        valid_loader=valid_loader)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
