import torch

from torch import nn
from torch.nn import functional as F

from pytorch_pretrained_bert import BertModel

from .similarity import get_similarity_func


class EncodingLayer(nn.Module):
    """
    Layer to encode a variable-length sentence as sum-pooling of a learned
    embedding.
    """

    def __init__(self, encoding_size, vocab):
        """
        Initialises the encoding layer.

        Parameters
        ---
        encoding_size : int
            Target size of the encoding.
        vocab : AbstractVocab
            Vocabulary used for the encodings.
        """
        super().__init__()

        self.vocab_size = len(vocab)
        self.padding_token_index = vocab.padding_token_index
        self.embeddings = vocab.name

        if self.embeddings == "bert":
            self.encoding_layer = BertModel.from_pretrained(
                'bert-base-uncased')
        else:
            self.encoding_layer = nn.Embedding(
                num_embeddings=self.vocab_size,
                embedding_dim=encoding_size,
                padding_idx=self.padding_token_index)

    def forward(self, sentences):
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
            T = sentences.shape[1]
            sen_length = sentences.shape[2]

        flattened = reshaped.reshape(-1, sen_length)

        if self.embeddings == "bert":
            # We don't want to fine-tune BERT!
            with torch.no_grad():
                encoded_layers, _ = self.encoding_layer(flattened)

            # We have a hidden states for each of the 12 layers
            # in model bert-base-uncased

            # Remove useless dimension
            pooled_flat = torch.squeeze(encoded_layers[11])
        else:
            encoded_flat = self.encoding_layer(flattened)
            pooled_flat = encoded_flat.sum(dim=1)

        # Re-shape into original form (4D or 3D tensor)
        enc_size = pooled_flat.shape[1]
        if N is not None and k is not None:
            encoded = pooled_flat.reshape(-1, N, k, enc_size)
        else:
            # Assume 3D tensor
            encoded = pooled_flat.reshape(-1, T, enc_size)

        return encoded


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
            # NOTE: The original paper concatenates both
            # to form a new state as [h, r], however other
            # implementations seem to just sum it, which
            # some dimensionality issues not clarified on the
            # paper.
            # Add r from last step
            h_prev += r_prev.view(-1, encoding_size)

            # Recurr on LSTM
            h_next, c_next = self.lstm_cell(flattened_targets,
                                            (h_prev, c_prev))
            h_next += flattened_targets

            # Unflat previous hidden state to compute attention
            attention = self._attention(
                h_prev.view(-1, T, encoding_size), support_embeddings)

            # Compute next value of r and re-flat
            r_next = torch.einsum('btnk,bnke->bte', attention,
                                  support_embeddings)
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
        attention : torch.Tensor[batch_size x T x N x k]
            Attention computed across pairs of targets and support sentences.
        """
        dot_products = torch.einsum('bte,bnke->btnk', h, support_embeddings)
        _, T, N, k = dot_products.shape
        flat_dot_products = dot_products.view(-1, T, N * k)
        flat_softmax = F.softmax(flat_dot_products, dim=2)
        return flat_softmax.view(-1, T, N, k)


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
                hidden_size=encoding_size,
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

        # Run LSTM across the support set of each episode, and separate
        # both directions (remember it's a BiLSTM layer)
        flattened_fce_encodings, _ = self.fce_layer(flattened_encodings)
        dir_fce_encodings = flattened_fce_encodings.view(
            -1, N, k, 2, encoding_size)

        # Sum both directions and the oringinal encoding
        # as per the paper
        fce_encodings = dir_fce_encodings.sum(dim=3) + support_encodings

        return fce_encodings


class MatchingNetwork(nn.Module):
    """
    Main model which uses all of the above.
    """

    def __init__(self,
                 name,
                 vocab,
                 fce=True,
                 processing_steps=5,
                 distance_metric="cosine"):
        """
        Initialises the Matching Network.

        Parameters
        ---
        name : str
            Name of the model. Used for storing checkpoints.
        vocab : AbstractVocab
            AbstractVocab object.
        fce : bool
            Flag to decide if we should use Full Context Embeddings.
        processing_steps : int
            How many processing steps to take when embedding
            the target query.
        distance_metric : str
            Distance metric to be used.
        """
        super().__init__()

        self.name = name

        self.encoding_size = 64
        if vocab.name == 'bert':
            self.encoding_size = 210

        self.vocab_size = len(vocab)

        self.encode = EncodingLayer(self.encoding_size, vocab)
        self.g = GLayer(self.encoding_size, fce=fce)
        self.f = FLayer(self.encoding_size, processing_steps=processing_steps)

        self.distance_metric = distance_metric

    def _similarity(self, support_embeddings, target_embeddings):
        """
        Takes a measure of similarity by using the cosine distance.

        Parameters
        ---
        support_embeddings : torch.Tensor[batch_size x N x k x encoding_size]
            Embeddings of the support set.
        target_embeddings : torch.Tensor[batch_size x T x encoding_size]
            Embeddings of the target set.

        Returns
        ---
        similarity : torch.Tensor[batch_size x T x N x k]
            Similarity of each target to each example in the support set.
        """
        batch_size, N, k, _ = support_embeddings.shape
        T = target_embeddings.shape[1]
        similarities = torch.zeros(batch_size, T, N, k)
        similarity_func = get_similarity_func(self.distance_metric)

        # TODO: Would be good to optimise this so that it's vectorised.

        # Compute similarity for each triple target/label/example
        for t_idx in range(T):
            # Extract targets at postition t_idx
            target_embeddings_t = target_embeddings[:, t_idx, :]

            for n_idx in range(N):
                for k_idx in range(k):
                    # Extract support embedding for label n and
                    # example k
                    support_embeddings_nk = support_embeddings[:, n_idx,
                                                               k_idx, :]

                    # Compute mean similarity with labels at n
                    similarities[:, t_idx, n_idx, k_idx] = similarity_func(
                        support_embeddings_nk, target_embeddings_t)

        return similarities

    def _attention(self, support_embeddings, target_embeddings):
        """
        Compute attention to each example on the support set.

        Parameters
        ---
        support_embeddings : torch.Tensor[batch_size x N x k x encoding_size]
            Embeddings of the support set.
        target_embeddings : torch.Tensor[batch_size x T x encoding_size]
            Embeddings of the target set.

        Returns
        ---
        attention : torch.Tensor[batch_size x T x N x k]
            Attention of each target to each label in the support set.
        """
        similarities = self._similarity(support_embeddings, target_embeddings)

        # Compute attention as a softmax over similarities
        _, T, N, k = similarities.shape
        flat_similarities = similarities.view(-1, T, N * k)
        flat_attention = F.softmax(flat_similarities, dim=2)
        attention = flat_attention.view(-1, T, N, k)

        return attention

    def _to_logits(self, attention, labels):
        """
        Convert attention to logits over entire vocabulary.

        Parameters
        ---
        attention : torch.Tensor[batch_size x T x N x k]
            Attention of each target to each label in the support set.
        labels : torch.Tensor[batch_size x N]
            Corresponding token of each label N in the vocabulary.

        Returns
        ---
        logits : torch.Tensor[batch_size x T x vocab_size]
            Predicted probabilities for each target of each episode.
        """
        # Sum across labels
        attention = attention.sum(dim=3)
        batch_size, T, N = attention.shape
        logits = torch.zeros((batch_size, T, self.vocab_size))

        # TODO: Would be good to optimise this so that it's vectorised.
        for batch_idx in range(batch_size):
            for t_idx in range(T):
                for l_idx, label_token in enumerate(labels[batch_idx]):
                    logits[batch_idx, t_idx, label_token] = attention[
                        batch_idx, t_idx, l_idx]

        return logits

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
        labels : torch.Tensor[batch_size x N]
            Corresponding token of each label N in the vocabulary.

        Returns
        ---
        logits : torch.Tensor[batch_size x T x vocab_size]
            Predicted probabilities for each target of each episode.
        """
        support_set, targets, labels = input_args

        # Encode both sets
        support_encodings = self.encode(support_set)
        target_encodings = self.encode(targets)

        # Embed both sets using f() and g()
        support_embeddings = self.g(support_encodings)
        target_embeddings = self.f(target_encodings, support_embeddings)

        # Compute attention matrix between support and target embeddings
        attention = self._attention(support_embeddings, target_embeddings)

        # Convert attention to logits over the entire vocabulary
        logits = self._to_logits(attention, labels)
        return logits
