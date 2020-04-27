import torch
import torch.nn as nn
import numpy as np


class VAE(nn.Module):
    """Sequence level VAE."""

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        latent_dim,
        max_seq_len,
        num_layers,
        rnn_type,
        device,
        embedding_dropout,
        word_dropout,
        bidirectional,
        pad_id,
        unk_id,
        sos_id,
        eos_id,
    ):
        super(VAE, self).__init__()

        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.device = device
        self.word_dropout = word_dropout
        self.hidden_dim = hidden_dim
        self.seq_len = max_seq_len
        self.pad_id = pad_id
        self.unk_id = unk_id
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.rnn_type = rnn_type
        # Set rnn type
        if rnn_type == "rnn":
            rnn = nn.RNN
        elif rnn_type == "gru":
            rnn = nn.GRU
        elif rnn_type == "lstm":
            rnn = nn.LSTM
        else:
            raise ValueError("Check rnn type.")

        self.hidden_factor = (2 if bidirectional else 1) * self.num_layers

        # Input
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        self.encoder_rnn = rnn(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
        )

        # Latent
        if self.rnn_type == "lstm":
            self.hidden_to_mean = nn.Linear(
                hidden_dim * self.hidden_factor * 2, latent_dim
            )
            self.hidden_to_logv = nn.Linear(
                hidden_dim * self.hidden_factor * 2, latent_dim
            )
        else:
            self.hidden_to_mean = nn.Linear(hidden_dim * self.hidden_factor, latent_dim)
            self.hidden_to_logv = nn.Linear(hidden_dim * self.hidden_factor, latent_dim)

        self.decoder_rnn = rnn(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
        )
        # Create intermediate and out linear layers
        if rnn_type == "lstm":
            self.latent_to_hidden = nn.Linear(
                latent_dim, hidden_dim * self.hidden_factor * 2
            )
            self.output_to_vocab = nn.Linear(
                hidden_dim * (2 if bidirectional else 1), vocab_size
            )
        else:
            self.latent_to_hidden = nn.Linear(
                latent_dim, hidden_dim * self.hidden_factor
            )
            self.output_to_vocab = nn.Linear(
                hidden_dim * (2 if bidirectional else 1), vocab_size
            )

    def forward(self, text, length):
        """
        Parameters
        ----------
        seq : 
            batch_size x seq_len
        """
        batch_size = text.size(0)
        sorted_lengths, sorted_idx = torch.sort(length, descending=True)
        text = text[sorted_idx]
        embedding = self.embedding(text)
        packed_input = nn.utils.rnn.pack_padded_sequence(
            embedding, sorted_lengths.tolist(), batch_first=True
        )
        if self.rnn_type == "lstm":
            _, (hidden, cell) = self.encoder_rnn(packed_input)
        else:
            _, hidden = self.encoder_rnn(packed_input)

        # Take last hidden state
        if self.bidirectional or self.num_layers > 1:
            if self.rnn_type == "lstm":
                hidden = hidden.view(batch_size, self.hidden_dim * self.hidden_factor)
                cell = cell.view(batch_size, self.hidden_dim * self.hidden_factor)
                hidden = torch.cat((hidden, cell), dim=-1)
            else:
                hidden = hidden.view(batch_size, self.hidden_dim * self.hidden_factor)
        else:
            if self.rnn_type == "lstm":
                hidden = hidden.squeeze()
                cell = cell.squeeze()
                hidden = torch.cat((hidden, cell,), dim=-1)
            else:
                hidden = hidden.squeeze()

        # Get latent variables
        mean = self.hidden_to_mean(hidden)
        logv = self.hidden_to_logv(hidden)
        std = torch.exp(0.5 * logv)
        z = torch.randn([batch_size, self.latent_dim], device=self.device)
        z = z * std + mean

        # Decode
        hidden = self.latent_to_hidden(z)
        if self.bidirectional or self.num_layers > 1:
            if self.rnn_type == "lstm":
                hidden = hidden.view(
                    self.hidden_factor, batch_size, self.hidden_dim * 2
                )
                cell = hidden[:, :, self.hidden_dim :]
                hidden = hidden[:, :, : self.hidden_dim]
                # cell = cell.view(self.hidden_factor, batch_size, self.hidden_dim)
            else:
                hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_dim)
        else:
            if self.rnn_type == "lstm":
                cell = hidden[:, self.hidden_dim :]
                hidden = hidden[:, : self.hidden_dim]
                cell = cell.unsqueeze(0)
            hidden = hidden.unsqueeze(0)
        # As per the authors normal dropout didn't help the decoder use z
        # hence use word dropout
        # embedding = self.decoder_dropout(embedding)
        # Generate random probabilites to dropout words
        word_drop_probability = torch.rand(text.size(), device=self.device)
        # Retain prob=1 fro sos and pad
        word_drop_probability[text - self.sos_id == 0] = 1
        word_drop_probability[text - self.pad_id == 0] = 1
        decoder_input = text.clone()
        decoder_input[word_drop_probability < self.word_dropout] = self.unk_id
        decoder_embedding = self.embedding(decoder_input)
        decoder_embedding = self.embedding_dropout(decoder_embedding)
        packed_input = nn.utils.rnn.pack_padded_sequence(
            decoder_embedding, sorted_lengths.tolist(), batch_first=True
        )
        #  Decoder Forward
        if self.rnn_type == "lstm":
            output, _ = self.decoder_rnn(
                packed_input, (hidden.contiguous(), cell.contiguous())
            )
        else:
            output, _ = self.decoder_rnn(packed_input, hidden)

        padded_output = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)[0]
        padded_output = padded_output.contiguous()
        _, reversed_id = torch.sort(sorted_idx)
        padded_output = padded_output[reversed_id]

        b, s, _ = padded_output.size()
        # Output to Vocab
        logp = nn.functional.log_softmax(
            self.output_to_vocab(padded_output.view(-1, padded_output.size(2))), dim=-1
        )
        logp = logp.view(b, s, self.embedding.num_embeddings)
        return logp, mean, logv, z

    def infer_single(self, n=4, z=None):
        for i in range(n):
            batch_size = 1
            if z is None:
                z = torch.randn([1, self.latent_dim]).to(self.device)
            hidden = self.latent_to_hidden(z)
            if self.bidirectional or self.num_layers > 1:
                hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_dim)
            else:
                hidden = hidden.unsqueeze(0)
            t = 0
            end = False
            input_seq = (
                torch.Tensor(batch_size).long().fill_(self.sos_id).to(self.device)
            )
            generated_sequence = (
                torch.Tensor(self.seq_len).long().fill_(self.pad_id).to(self.device)
            )
            while t < self.seq_len and not end:
                if len(input_seq.size()) == 0:
                    input_seq = input_seq.unsqueeze(0)
                input_seq = input_seq.unsqueeze(-1)
                # print("Input Seq Size Unqueezes: ", input_seq.size())
                # print("="*84)

                input_embedding = self.embedding(input_seq)
                # print("Input Embedding: ", input_embedding)
                # print("Input Embedding Shape:", input_embedding.size())
                # print("="*84)
                output, hidden = self.decoder_rnn(input_embedding, hidden)
                # print("Output Size: ", output.size())
                logits = self.output_to_vocab(output)
                # print(logits.size())
                # print("="*84)
                _, sample = torch.topk(logits, 1, dim=-1)
                # print("Sample sizze: ", sample.size())
                input_seq = sample.squeeze()
                generated_sequence[t] = input_seq.item()
                if input_seq.item() == self.eos_id:
                    break
                t += 1
            print("Generate Sentence: ", generated_sequence)

    def infer(self, n=4, z=None, strategy="greedy"):
        # TODO: Use Beam Search for Inference
        # Create hidden
        if z is None:
            batch_size = n
            z = torch.randn([batch_size, self.latent_dim]).to(self.device)
        else:
            batch_size = z.size(0)

        # Decode
        hidden = self.latent_to_hidden(z)
        if self.bidirectional or self.num_layers > 1:
            if self.rnn_type == "lstm":
                hidden = hidden.view(
                    self.hidden_factor, batch_size, self.hidden_dim * 2
                )
                cell = hidden[:, :, self.hidden_dim :]
                hidden = hidden[:, :, : self.hidden_dim]
            else:
                hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_dim)
        else:
            if self.rnn_type == "lstm":
                cell = hidden[:, self.hidden_dim :]
                hidden = hidden[:, : self.hidden_dim]
                cell = cell.unsqueeze(0)
            hidden = hidden.unsqueeze(0)

        sequence_id = torch.arange(0, batch_size).long().to(self.device)
        sequence_mask = torch.ones(batch_size).type(torch.bool).to(self.device)
        current_sequence = torch.arange(0, batch_size).long().to(self.device)
        remaining_sequence = torch.arange(0, batch_size).long().to(self.device)
        generated_sequences = (
            torch.Tensor(batch_size, self.seq_len)
            .long()
            .fill_(self.pad_id)
            .to(self.device)
        )
        generated_scores = torch.tensor(batch_size).float().fill_(0.0).to(self.device)
        t = 0
        while t < self.seq_len and len(remaining_sequence) > 0:
            if t == 0:
                input_seq = (
                    torch.Tensor(batch_size).long().fill_(self.sos_id).to(self.device)
                )

            if len(remaining_sequence) == 1 and len(input_seq.size()) == 0:
                input_seq = input_seq.unsqueeze(0)
            # print("Input Seq: ", input_seq)
            print("Input Seq Size: ", input_seq.size())
            input_seq = input_seq.unsqueeze(-1)

            input_embedding = self.embedding(input_seq)
            if self.rnn_type == "lstm":
                output, (hidden, cell) = self.decoder_rnn(
                    input_embedding, (hidden.contiguous(), cell.contiguous())
                )
            else:
                output, hidden = self.decoder_rnn(input_embedding, hidden)

            logits = self.output_to_vocab(output)  # logits: batch_size x vocab_size
            sample = self.sample_next_word(logits, strategy=strategy)
            # torch.topk(logits, 1, dim=-1)
            print("Sample size: ", sample.size())
            input_seq = sample.squeeze()  # Sample: batch_size x _ x k
            # Save the next word
            # batch_size x seq_len
            generated_sequences = self.save_sample(
                generated_sequences, input_seq, current_sequence, t
            )
            sequence_mask[current_sequence] = (
                input_seq != self.eos_id
            )  # sequence_mask: batch_size
            # print("seq mask: ", sequence_mask)
            current_sequence = sequence_id.masked_select(sequence_mask)
            # print("curent seq:", current_sequence) # current_sequence: batch_size
            remaining_mask = input_seq != self.eos_id  # remaining_mask: batch_size
            # print("remain mask: ", remaining_mask)
            # batch_size
            remaining_sequence = remaining_sequence.masked_select(remaining_mask)
            # print("remain seq: ", remaining_sequence)
            if len(remaining_sequence) > 0:
                if len(remaining_sequence) == 1 and len(input_seq.size()) == 0:
                    pass
                else:
                    # batch_size
                    input_seq = input_seq[remaining_sequence]
                    # print("="*84)
                    # print("New input seq: ", input_seq)
                    hidden = hidden[:, remaining_sequence]
                    if self.rnn_type == "lstm":
                        cell = cell[:, remaining_sequence]
                # print("Old remaining seq: ", remaining_sequence)
                # batch_size
                remaining_sequence = (
                    torch.arange(0, len(remaining_sequence)).long().to(self.device)
                )
                # print("New remaining seq: ", remaining_sequence)
            t += 1

        return generated_sequences, z

    def sample_next_word(self, logits, strategy="greedy"):
        if strategy == "greedy":
            _, sample = torch.topk(logits, 1, dim=-1)
            return sample
        elif strategy == "random":
            sample = torch.randint(
                low=0, high=logits.size(-1), size=(logits.size(0), 1)
            ).to(self.device)
            return sample

    def save_sample(self, save_to, sample, current_sequence, t):
        current = save_to[current_sequence]
        current[:, t] = sample
        save_to[current_sequence] = current
        # batch_size x seq_len
        print("current save: ", current)
        return save_to

    def beam_search(self, batch_size=4, max_seq_len=60, beam_width=3, z=None):
        # input_sequence: batch_size x seq_len
        # logits: batch_size x vocab
        # cadidate_scores: batch_size x beam_width
        # beam_width: int
        # decoder: takes in input and give out logits # assume
        if z is None:
            z = torch.randn([batch_size, self.latent_dim]).to(self.device)
        else:
            batch_size = z.size(0)

        hidden = self.latent_to_hidden(z)
        if self.bidirectional or self.num_layers > 1:
            if self.rnn_type == "lstm":
                hidden = hidden.view(
                    self.hidden_factor, batch_size, self.hidden_dim * 2
                )
                cell = hidden[:, :, self.hidden_dim :]
                hidden = hidden[:, :, : self.hidden_dim]
            else:
                hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_dim)
        else:
            if self.rnn_type == "lstm":
                cell = hidden[:, self.hidden_dim :]
                hidden = hidden[:, : self.hidden_dim]
                cell = cell.unsqueeze(0)
            hidden = hidden.unsqueeze(0)
        sequences = torch.zeros(
            (batch_size, beam_width, max_seq_len+1), device=self.device
        ).long().fill_(self.sos_id)
        candidate_scores = (
            torch.zeros((batch_size, beam_width), device=self.device)
            .float()
        )
    
        if self.rnn_type == 'lstm':
            cell = self.tile(cell, 1, beam_width)
        hidden = self.tile(hidden, 1, beam_width)
        t = 0
        while t < max_seq_len:
            # print("Start: ", sequences.size())
            #sequences = sequences.view(-1, max_seq_len)
            prev_word = sequences[:, :, t]
            t += 1
            prev_word = prev_word.view(-1, 1)
            # print("Change View: ", prev_word.size())
            decoder_input_embedding = self.embedding(prev_word)
            # print("Input Embedding: ", decoder_input_embedding.size())
            if self.rnn_type == "lstm":
                output, (hidden, cell) = self.decoder_rnn(
                    decoder_input_embedding, (hidden.contiguous(), cell.contiguous())
                )
            else:
                output, hidden = self.decoder_rnn(decoder_input_embedding, hidden)
            logits = self.output_to_vocab(output)
            # logits = decoder(sequences.view(-1, max_seq_len))
            # (batch_size x beam_width) x max_seq_len
            logprobs = torch.log_softmax(logits, dim=-1)
            # print("Logprob: ", logprobs.size())
            values, indices = torch.topk(logprobs, dim=-1, k=beam_width)
            # print("Values", values.size())
            # (batch_size x beam_width) x beam_width
            values = values.view(batch_size, beam_width * beam_width)

            candidate_scores_tiled = self.tile(candidate_scores, 1, beam_width)
            # batch_size x (beam_width x beam_width)
            values_to_select_from = candidate_scores_tiled + values
            # batch_size x (beam_width x beam_width)
            candidate_values, candidate_indices = torch.topk(
                values_to_select_from, beam_width, dim=-1
            )
            # batch_size x beam_width

            next_word_to_select = candidate_indices % beam_width
            # batch_size x beam_width
            sequence_to_select = candidate_indices // beam_width
            # batch_size x beam_width
            candidate_scores = candidate_values

            values = values.view(batch_size, beam_width, beam_width)
            indices = indices.view(batch_size, beam_width, beam_width)
            for b in range(batch_size):
                for w in range(beam_width):
                    # print("word to fill: ", sequences[b, sequence_to_select[b, w], t])
                    # print("fill this: ", indices[b, sequence_to_select[b, w], next_word_to_select[b, w]])
                    sequences[b, sequence_to_select[b, w], t] = indices[b, sequence_to_select[b, w], next_word_to_select[b, w]]
                    sequences[b, w] = sequences[b, sequence_to_select[b, w]]
        #print(sequences.size())
        vals, inds = torch.topk(candidate_scores, dim=-1, k=1)
        final_sequences = torch.zeros((batch_size, max_seq_len+1), device=self.device)
        for b in range(batch_size):
            final_sequences[b] = sequences[b, inds[b]]
        #return sequences.view(-1, max_seq_len+1)
        return final_sequences

    def tile(self, a, dim, n_tile):
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx))
        order_index = torch.tensor(
            np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).long().to(self.device)
        return torch.index_select(a, dim, order_index)
