import torch

from torch import nn
from torch.autograd import Variable

from .process import VOCAB_SIZE


class SentenceEncoder(nn.Module):
    """
    Encoder to transform the input sentences into embeddings, by applying a
    linear layer and then (optionally) FCE.
    """

    def __init__(self, vocab_size, layer_size=64, fce=False):
        """
        Initialise the sentence encoder.

        Parameters
        ---
        vocab_size : int
            Vocabulary length. Used to transform the values into
            one-hot vectors.
        layer_size : int
            Size of the final embedding.
        fce : bool
            Flag to use Full Context Embeddings.
        """
        super(SentenceEncoder, self).__init__()

        self.vocab_size = vocab_size
        self.layer_size = layer_size
        self.out_size = layer_size
        self.fce = fce

        # Linear layer
        self.layer1 = nn.Linear(self.vocab_size, self.layer_size)

        if self.fce:
            self.lstm = BidirectionalLSTM(
                layer_size=32, vector_dim=self.out_size)

    def forward(self, support_set):
        """
        Encodes the sentences on the support set.

        Parameters
        ----
        support_set : torch.Tensor[batch_size x N x k x sen_length]
            Support set containing [batch_size] episodes of [N] labels
            with [k] examples each. The last dimension represents the
            encoding of each sentence.

        Returns
        ----
        embedded_set : torch.Tensor[batch_size x N x k x ]
            Encoded sentences as one-hot vectors.
        """
        embedded = self.layer1(
            Variable(torch.from_numpy(
                support_set.transpose())).type('torch.FloatTensor'))

        if self.fce:
            fce_embedded = self.lstm(embedded)
            return fce_embedded

        return embedded


class DistanceNetwork(nn.Module):
    """
    This model calculates the cosine distance between each of the support set
    embeddings and the target embeddings.
    """

    def forward(self, support_set, targets):
        """
        Forward layer of the distance network.

        Parameters
        ---
        support_set : torch.Tensor[batch_size x N x k x emb_size]
            Embedded support set.
        targets : torch.Tensor[batch_size x T x emb_size]
            Embedded set of targets to predict.

        Returns
        ---
        similarities : torch.Tensor[batch_size x T x N]
            Similarities to each one of the labels.
        """
        # TODO: Fix this, doesn't work with batches
        # Also, it's not a network, should be just a generic distance function
        # instead.

        eps = 1e-10
        similarities = []
        for support_image in support_set:
            sum_support = torch.sum(torch.pow(support_image, 2), 1)
            support_manitude = sum_support.clamp(eps, float("inf")).rsqrt()
            dot_product = input_image.unsqueeze(1).bmm(
                support_image.unsqueeze(2)).squeeze()
            cosine_similarity = dot_product * support_manitude
            similarities.append(cosine_similarity)
        similarities = torch.stack(similarities)
        return similarities.t()


class BidirectionalLSTM(nn.Module):
    """
    FCE optional encoder with one layer of bidirectional LSTMs.
    """

    def __init__(self, layer_size, vector_dim):
        super(BidirectionalLSTM, self).__init__()
        """
        Initialise the BiLSTM.

        Parameters
        ---
        layer_size : int
            Number of LSTM units (hidden size).
        vector_dim : int
            Size of the input.
        """
        self.hidden_size = layer_size
        self.vector_dim = vector_dim

        self.lstm = nn.LSTM(
            input_size=self.vector_dim,
            hidden_size=self.hidden_size,
            bidirectional=True)

        self.hidden = self._init_hidden()

    def _init_hidden(self):
        h0 = Variable(
            torch.zeros(2, self.lstm.hidden_size), requires_grad=False)
        c0 = Variable(
            torch.zeros(2, self.lstm.hidden_size), requires_grad=False)
        return (h0, c0)

    def _repackage_hidden(self, h):
        """
        Wraps hidden states in new Variables, to detach them from their
        history.
        """
        if type(h) == Variable:
            return Variable(h.data)
        else:
            return tuple(self.repackage_hidden(v) for v in h)

    def forward(self, inputs):
        """
        Forward pass of the BiLSTM layer to encode the inputs
        using FCE.

        Parameters
        ----
        inputs : torch.Tensor[batch_size x N x k x vector_dim]
            Input to encode.

        Returns
        ---
        output : torch.Tensor[batch_size x N x k x vector_dim]
            Encoded output.
        """
        # self.hidden = self.init_hidden(self.use_cuda)
        # self.hidden = self.repackage_hidden(self.hidden)
        output, self.hidden = self.lstm(inputs, self.hidden)

        return output


class MatchingNetwork(nn.Module):
    """
    Main model which uses all of the above.
    """

    def __init__(self, fce=False, vocab_size=VOCAB_SIZE):
        """
        Initialises the Matching Network.

        Parameters
        ---
        fce : bool
            Flag to decide if we should use Full
            Context Embeddings.
        vocab_size : int
            Size of the vocabulary to do one-hot encodings.
        """
        super(MatchingNetwork, self).__init__()

        self.g = SentenceEncoder(vocab_size, layer_size=64, fce=fce)
        self.dn = DistanceNetwork()

    def forward(self, input_args):
        """
        Implementation of the forward pass of the main network.

        Parameters
        ---
        support_set : torch.Tensor[batch_size x N x k x
                                    sen_length x vocab_size]
            Support set containing [batch_size] episodes of [N] labels
            with [k] examples each. The last dimension represents the
            list of tokens in each sentence.
        targets : torch.Tensor[batch_size x T x sen_length x vocab_size]
            List of targets to predict.

        Returns
        ---
        predictions : torch.Tensor[batch_size x N x k]
            Predicted label for each example of each episode.
        """
        support_set, targets = input_args

        # produce embeddings for support set entries
        embeddings = self.g(support_set)

        # get similarities between support set embeddings and target
        # TODO: Fix dn
        #  similarites = self.dn(support_set=embeddings, input_image=targets)

        return embeddings
