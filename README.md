# Statistical NLP (Group 17)

This is the repository for Group 17 of the Statistical Natural Language
Processing module at UCL, formed by:

- Talip Ucar (talip.ucar.16@ucl.ac.uk)
- Adrian Daniel Szwarc (adrian.szwarc.18@ucl.ac.uk)
- Matthew Lee (matthew.lee.16@ucl.ac.uk)
- Adrian Gonzalez-Martin (adrian.martin.18@ucl.ac.uk)

This repository implements the Matching Networks architecture ([Vinyals et al.,
2016](http://arxiv.org/abs/1606.04080)) in `pytorch` and applies it to a
Language Modelling task.

The architecture is flexible enough to allow easy experimentation with distance
metrics, number of labels per episode, number of examples per label, etc.

More details can be found in the associated paper.

## Getting Started

To keep the environments as reproducible as possible, we will use `pipenv` to
handle dependencies. Feel free to use `conda` or other tools which manage global
deps as well!

To install `pipenv` just follow the instructions in https://pipenv.readthedocs.io/en/latest/.

The first time, to create the environment and install all required dependencies,
just run

```console
$ pipenv install
```

This will create a `virtualenv` and will install all required dependencies.

### Installing new dependencies

To add new dependencies just run

```console
$ pipenv install numpy
```

Remember to commit the updated `Pipfile` and `Pipfile.lock` files so that
everyone else can also install them!

## Folder Structure

Most of the source code can be found under the `src/` folder. However, we also
include a set of command line tools, which should help with sampling, training
and testing models. These can be found under the `bin/` folder.

Additionally, you can find the following folders:

- `wikitext-2/`: Raw WikiText-2 data set.
- `data/`: Pre-sampled set of label/sentence pairs and pre-generated vocabulary.
- `models/`: Pre-trained models. The filenames encode the different parameters
  used to train the model.
- `results/`: Data generated after evaluating the models. It includes
  predictions on the test set, embeddings and attention maps.
- `figures/`: Figures generated from the data in the `results/` folder.

## Tests

We are using `pytest` for writing and running unit tests. You can see some
examples on the `test/` folder.

To run all tests, just run the following command.

```console
$ pytest -s src/tests
```

## Dataset

On the `data/` folder you can find a `train.csv` and `test.csv` files, which
contain each 9000 labels with 10 examples each and 1000 labels with 10 examples
each respectively.

The data is in CSV format with two columns:

- `label`: The word acting as label which we need to find.
- `sentence`: The sentence acting as input, where the particular word has been
  replaced with the token `<blank_token>`.

An example can be seen below:

```csv
label,sentence
music,no need to be a hipster to play <blank_token> in vynils
music,nowadays <blank_token> doesn't sound as before
...
```

### Sampling new pairs

If you want to sample a new set pairs from the WikiText-2 dataset you can use
the `bin.sample` script. For example, to resample the entire dataset, we could
just run:

```console
$ python -m bin.sample -N 9000 -k 10 wikitext-2/wiki.train.tokens data/train.csv
$ python -m bin.sample -N 1000 -k 10 wikitext-2/wiki.test.tokens data/test.csv
```

Note that the file will be processed first, to be as similar as text coming from
PTB.

### Generating vocabulary

To make things easy to replicate, we generate in advance the vocabulary over the
training set and store it in a file, which can then be used later for training
and testing. You can have a look at the format in `data/vocab.json`.

To re-generate it (after sampling new pairs, for example), you can use the
`bin.vocab` script:

```console
$ python -m bin.vocab data/train.csv data/vocab.json
```

This command will store the vocabulary's state as a `JSON` file.

## Training

Training of a new model can be performed using the `bin.train` script.

```console
$ python -m bin.train -N 5 -k 2 -e euclidean data/vocab.json data/train.csv
```

The `N` and `k` parameters control the number of labels and examples we want per
episode respectively. The other parameters refer to other parameters (like
distance metric) and the pre-computed vocabulary and the training set.

After convergence, the best model's `state_dict` is stored under the `models/`
folder, with the different parameters encoded in its name. For example, the
model `poincare_vanilla_N=5_k=2_model_7.pth` was trained using the `poincare`
distance metric, `vanilla` embedding, using `5` labels with `2` examples each
per episode. From the file name it can also be seen that it converged after `7`
epochs.

These details are discussed in further detail in the associated paper.

## Evaluation

Accuracy on a test set for a given model's snapshot can be measured using the
`bin.test` script:

```console
$ python -m bin.test -v data/vocab.json -m models/euclidean_vanilla_N\=5_k\=3_model_24.pth data/test.csv
```

These command has extra flags which allow to:

- `-p`: Store the predictions in the `results/` folder.
- `-e`: Generate embeddings and attention for a single episode and store them in
  the `results/` folder.

Some of the already generated data can be seen in the `results/` folder.
