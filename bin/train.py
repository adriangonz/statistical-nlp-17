"""
Command to train the model.
"""
from argparse import ArgumentParser

from torch.utils.data import DataLoader

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
parser.add_argument("training_set", help="Path to the training set")


def main(args):
    print("Initialising dataset...")
    data_set = EpisodesDataset(file_path=args.training_set, k=args.k)
    episodes_sampler = EpisodesSampler(data_set, N=args.N)

    print("Initialising dataloader...")
    train_loader = DataLoader(
        data_set, sampler=episodes_sampler, batch_size=BATCH_SIZE)

    print("Initialising model...")
    model = MatchingNetwork(fce=True, processing_steps=3)

    print("Starting to train...")
    train(model, learning_rate=1e-3, train_loader=train_loader)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
