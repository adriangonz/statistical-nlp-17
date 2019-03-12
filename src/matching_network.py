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
    Layer responsible of finding embeddings on the target set.

    Referred to in the paper as `f(x)` on the paper.
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
        super().__init__()
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
        for step in range(self.processing_steps):
            # Add r from last step
            h_prev += r_prev.view(-1, encoding_size)

            # Recurr on LSTM
            h_next, c_next = self.lstm_cell(flattened_targets,
                                            (h_prev, c_prev))
            h_next += flattened_targets

            # Unflat previous hidden state to compute attention
            attention = self._attention(
                h_prev.view(-1, T, encoding_size), support_embeddings)

            # Compute next value of r
            r_next = torch.sum(attention * support_embeddings, dim=2)
            r_next = r_next.view(-1, encoding_size)

            # Forward current state
            h_prev = h_next
            c_prev = c_next
            r_prev = r_next

        # Take the last h_prev and un-flat
        return h_prev.view(-1, T, encoding_size)

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
        # TODO: If something does not work, check this!!
        dot_products = torch.einsum('bte,bnke->bnke', h, support_embeddings)
        return F.softmax(dot_products, dim=3)


class GLayer(nn.Module):
    """
    Layer responsible of finding embeddings of the support set.

    Referred to in the paper as `g()`.
    """

    def __init__(self, encoding_size, fce):
        """
        Initialise the g()-layer.

        Parameters
        ---
        encoding_size : int
            Size of the sentence encodings.
        fce : bool
            Flag to decide if we should use Full Context Embeddings.
        """
        super().__init__()

        self.fce_layer = None
        if fce:
            self.fce_layer = nn.LSTM(
                input_size=encoding_size,
                hidden_size=encoding_size // 2,
                bidirectional=True,
                batch_first=True)

    def forward(self, support_encodings):
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
        if self.fce_layer is None:
            return support_encodings

        # Flatten encodings first so that the shape
        # is [batch_size x (N*k) x encoding_size] and the
        # support set entries are considered as a sequence
        _, N, k, encoding_size = support_encodings.shape
        flattened_encodings = support_encodings.view(-1, N * k, encoding_size)

        # Run LSTM across the support set of each episode
        flattened_fce_encodings, _ = self.fce_layer(flattened_encodings)

        # Un-flat output
        fce_encodings = flattened_fce_encodings.view(-1, N, k, encoding_size)
        return fce_encodings


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
            Flag to decide if we should use Full Context Embeddings.
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

        self.g = GLayer(encoding_size, fce=fce)
        self.f = FLayer(
            encoding_size=encoding_size, processing_steps=processing_steps)

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

        # Encode both sets
        support_encodings = self._encode(support_set)
        target_encodings = self._encode(targets)

        # Embed both sets using f() and g()
        support_embeddings = self.g(support_encodings)
        target_embeddings = self.f(target_encodings, support_embeddings)

        import ipdb
        ipdb.set_trace()
        # get similarities between support set embeddings and target
        # TODO: Fix dn
        #  similarites = self.dn(support_set=embeddings, input_image=targets)

        return target_embeddings
