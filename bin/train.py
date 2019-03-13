"""
Command to train the model.
"""
from argparse import ArgumentParser

from torch.utils.data import DataLoader

from src.data import read_vocab, read_data_set
from src.datasets import EpisodesSampler, EpisodesDataset
from src.matching_network import MatchingNetwork
from src.training import train

BATCH_SIZE = 3

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


def main(args):
    print("Loading dataset...")
    vocab = read_vocab(args.vocab)
    X_train, y_train = read_data_set(args.training_set, vocab)
    data_set = EpisodesDataset(X_train, y_train, k=args.k)

    print("Initialising model...")
    episodes_sampler = EpisodesSampler(data_set, N=args.N)
    train_loader = DataLoader(
        data_set, sampler=episodes_sampler, batch_size=BATCH_SIZE)
    model = MatchingNetwork(fce=True, processing_steps=3)

    print("Starting to train...")
    train(model, learning_rate=1e-3, train_loader=train_loader)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
