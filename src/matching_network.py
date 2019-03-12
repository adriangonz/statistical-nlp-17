import torch

from torch import nn
from torch.nn import functional as F

from .process import VOCAB_SIZE


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


class FLayer(nn.Module):
    """
    Layer to find embeddings on the target set.

    Referred to as `f(x)` on the paper.
    """

    def __init__(self, encoding_size, processing_steps):
        """
        Initialise the `f()` layer.

        Parameters
        ----
        encoding_size : int
            Size of the sentence encodings.
        processing_steps : int
            Number of processing steps for the LSTM.
            Referred to as `K` on the paper.
        """
        self.lstm_cell = nn.LSTMCell(
            input_size=encoding_size, hidden_size=encoding_size)
        self.processing_steps = processing_steps

    def forward(self, targets, support_embeddings):
        """
        Find an embedding of the targets.

        Parameters
        ----
        targets : torch.Tensor[batch_size x T x encoding_size]
            List of targets to predict.
        support_embeddings : torch.Tensor[batch_size x N x k x encoding_size]
            Embeddings of the support set.

        Returns
        ----
        embeddings : torch.Tensor[batch_size x T x encoding_size]
        """
        # Flatten so that targets are 2D
        # (i.e. [(batch_size * T) x encoding_size])
        T = targets.shape[1]
        encoding_size = targets.shape[2]
        flattened_targets = targets.view(-1, encoding_size)

        h_prev = torch.zeros_like(flattened_targets)
        c_prev = torch.zeros_like(flattened_targets)
        r_prev = torch.zeros_like(flattened_targets)
        for step in self.processing_steps:
            h_next, c_next = self.lstm_cell(flattened_targets, h_prev, c_prev)
            h_next += flattened_targets

            # Unflat previous hidden state to compute attention
            attention = self._attention(
                h_prev.view(-1, T, encoding_size), support_embeddings)

            # Compute next state and flatten
            r_next = torch.sum(attention * support_embeddings, axis=(2, 3))
            h_next = r_next.view(-1, encoding_size) + h_next

            # Forward current state
            h_prev = h_next
            c_prev = c_next

            import ipdb
            ipdb.set_trace()

    def _attention(self, h, support_embeddings):
        """
        Compute attention between the hidden states across
        all elements and the support embeddings.

        Parameters
        ---
        h : torch.Tensor[batch_size x T x encoding_size]
            Hidden states across the flattened targets across
            episodes and batches.
        support_embeddings : torch.Tensor[batch_size x N x k x encoding_size]
            Support embeddings.

        Returns
        ---
        attention : torch.Tensor[batch_size x N x k x encoding_size]
            Attention computed across pairs of targets and support sentences.
        """
        dot_products = torch.einsum('bte,bnke->bnke', h, support_embeddings)
        return F.softmax(dot_products)


class MatchingNetwork(nn.Module):
    """
    Main model which uses all of the above.
    """

    def __init__(self, fce=False, vocab_size=VOCAB_SIZE, processing_steps=1):
        """
        Initialises the Matching Network.

        Parameters
        ---
        fce : bool
            Flag to decide if we should use Full
            Context Embeddings.
        vocab_size : int
            Size of the vocabulary to do one-hot encodings.
        processing_steps : int
            How many processing steps to take when embedding
            the target query.
        """
        super().__init__()

        encoding_size = 64

        self.encoding_layer = nn.EmbeddingBag(
            num_embeddings=vocab_size, embedding_dim=encoding_size, mode='sum')

        if fce:
            # The hidden size will half the encoding,
            # because the LSTM is bidirectional
            self.g_layer = nn.LSTM(
                input_size=encoding_size,
                hidden_size=encoding_size // 2,
                bidirectional=True,
                batch_first=True)

        self.processing_steps = processing_steps
        self.f = FLayer(encoding_size=encoding_size, processing_steps=1)

        #  self.dn = DistanceNetwork()

    def _encode(self, sentences):
        """
        Encode a set of sentences.

        Parameters
        ---
        sentences : torch.Tensor[batch_size x (N x k | T) x sen_length]
            Sentences to encode. The shapes can be variable.

        Returns
        ---
        encodings : torch.Tensor[batch_size x (N x k | T) x encoding_size]
        """
        # Save original shape to reshape afterwards
        N = k = T = sen_length = None

        # Reshape into 3D Tensor and flatten
        reshaped = sentences
        if len(sentences.shape) == 4:
            N = sentences.shape[1]
            k = sentences.shape[2]
            sen_length = sentences.shape[3]
            reshaped = sentences.view(-1, N * k, sen_length)
        else:
            # Assume 3D tensor
            sen_length = sentences.shape[2]
            T = sentences.shape[1]

        flattened = reshaped.reshape(-1, sen_length)

        # TODO: Work out how to remove padding
        encoded_flat = self.encoding_layer(flattened)

        # Re-shape into original form (4D or 3D tensor)
        enc_size = encoded_flat.shape[1]
        if N is not None and k is not None:
            encoded = encoded_flat.reshape(-1, N, k, enc_size)
        else:
            # Assume 3D tensor
            encoded = encoded_flat.reshape(-1, T, enc_size)

        return encoded

    def _g(self, support_encodings):
        """
        Find an embedding of the support set.

        Parameters
        ---
        support_set : torch.Tensor[batch_size x N x k x encoding_size]
            Support set containing [batch_size] episodes of [N] labels
            with [k] examples each. The last dimension represents the
            list of tokens in each sentence.

        Returns
        ---
        embeddings : torch.Tensor[batch_size x N x k x encoding_size]
            Embeddings.
        """
        if not self.fce:
            return support_encodings

        return self.g_layer(support_encodings)

    def forward(self, input_args):
        """
        Implementation of the forward pass of the main network.

        Parameters
        ---
        support_set : torch.Tensor[batch_size x N x k x sen_length]
            Support set containing [batch_size] episodes of [N] labels
            with [k] examples each. The last dimension represents the
            list of tokens in each sentence.
        targets : torch.Tensor[batch_size x T x sen_length]
            List of targets to predict.

        Returns
        ---
        predictions : torch.Tensor[batch_size x N x k]
            Predicted label for each example of each episode.
        """
        support_set, targets = input_args

        # produce encodings for all entries
        support_encodings = self._encode(support_set)
        target_encodings = self._encode(targets)

        support_embeddings = self._g(support_encodings)
        f_embeddings = self.f(target_encodings, support_embeddings)

        # get similarities between support set embeddings and target
        # TODO: Fix dn
        #  similarites = self.dn(support_set=embeddings, input_image=targets)

        import ipdb
        ipdb.set_trace()
        return f_embeddings
