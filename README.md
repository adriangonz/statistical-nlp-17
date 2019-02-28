# Statistical NLP (Group 17)

This is the repository for Group 17 of the Statistical Natural Language
Processing module at UCL, formed by:

- Talip Ucar
- Adrian Swarzc
- Matt Lee
- Adrian Gonzalez-Martin

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

## Source Code

This repo should be treated as the single source of truth for source code,
therefore we should keep most of it as a library under the `src/` folder, so
that notebooks (or `ipython`, Colab, etc.) can just do something like

```python
from .src.models import MatchingNetworks
```

This way we avoid fragmentation across notebooks. This is also the most
platform-agnostic method as it's more generic and doesn't make any assumptions
on where the code will be run.

### Tests

We are using `pytest` for writing and running unit tests. You can see some
examples on the `test/` folder.

To run all tests, just type on the terminal

```console
$ pytest -s src/tests
```

## Dataset

On the `data/` folder you can find a `train.csv` and `test.csv` files, which
contain each 9000 labels with 10 examples each and 1000 with 10 examples each
respectively.

The data is in CSV format with two columns:

- `label` The word acting as label which we need to find.
- `sentence` The sentence acting as input, where the particular word has been
  replaced with the token `<blank_token>`.

An example can be seen below:

```csv
label,sentence
music,no need to be a hipster to play <blank_token> in vynils
music,nowadays <blank_token> doesn't sound as before
...
```

### Sampling new pairs

There is a script in the `bin` package which can be used to sample pairs of
sentences and missing words out of a WikiText-2 file. Note that the file will be
processed first, to be as similar as text coming from PTB.

As an example, to sample sample 9000 labels with 10 examples each we would run
the following

```console
$ python -m bin.sample -N 9000 -k 10 wikitext-2/wiki.train.tokens data/train.csv
```
