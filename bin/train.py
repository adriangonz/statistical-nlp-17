"""
Command to train the model.
"""
from argparse import ArgumentParser

from torch.utils.data import DataLoader

from src.data import read_vocab, read_data_set
from src.datasets import EpisodesSampler, EpisodesDataset
from src.matching_network import MatchingNetwork
from src.training import train
from src.utils import train_test_split_tensors, get_model_name

# Matching Networks's paper specifies 20
# as the batch size
BATCH_SIZE = 20

# Percentage of the training set used for
# validation
VAL_PERC = 0.1

LEARNING_RATE = 1e-3

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
parser.add_argument(
    "-d",
    "--distance-metric",
    action="store",
    dest="distance_metric",
    type=str,
    default='cosine',
    help="Distance metric to be used")
parser.add_argument(
    "-p",
    "--processing-steps",
    action="store",
    dest="processing_steps",
    type=int,
    default=5,
    help="Number of processing steps used for FCE")
parser.add_argument("vocab", help="Path to the vocab JSON file")
parser.add_argument("training_set", help="Path to the training CSV file")


def _get_loader(data_set, N, episodes_multiplier=1):
    sampler = EpisodesSampler(
        data_set, N=N, episodes_multiplier=episodes_multiplier)
    loader = DataLoader(data_set, sampler=sampler, batch_size=BATCH_SIZE)
    return loader


def main(args):
    print("Loading dataset...")
    vocab = read_vocab(args.vocab)
    X_train, y_train = read_data_set(args.training_set, vocab)

    # Split training further into train and valid
    X_train, X_valid, y_train, y_valid = train_test_split_tensors(
        X_train, y_train, test_size=VAL_PERC)
    train_set = EpisodesDataset(X_train, y_train, k=args.k)
    valid_set = EpisodesDataset(X_valid, y_valid, k=args.k)

    print("Initialising model...")
    model_name = get_model_name(
        distance=args.distance_metric,
        embeddings='vanilla',
        N=args.N,
        k=args.k)
    model = MatchingNetwork(
        model_name,
        fce=True,
        processing_steps=args.processing_steps,
        distance_metric=args.distance_metric)

    print("Starting to train...")
    train_loader = _get_loader(train_set, args.N)
    valid_loader = _get_loader(valid_set, args.N, episodes_multiplier=20)
    train(
        model,
        learning_rate=LEARNING_RATE,
        train_loader=train_loader,
        valid_loader=valid_loader)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
