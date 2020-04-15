import torch
import torch.nn as nn


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
        TEXT,
    ):
        super(VAE, self).__init__()

        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.device = device
        self.word_dropout = word_dropout
        self.hidden_dim = hidden_dim
        self.seq_len = max_seq_len
        self.TEXT = TEXT
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
        self.hidden_to_mean = nn.Linear(hidden_dim * self.hidden_factor, latent_dim)
        self.hidden_to_logv = nn.Linear(hidden_dim * self.hidden_factor, latent_dim)

        self.decoder_rnn = rnn(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
        )
        self.decoder_dropout = nn.Dropout(word_dropout)
        # Create intermediate and out linear layers
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim * self.hidden_factor)
        self.output_to_vocab = nn.Linear(
            hidden_dim * (2 if bidirectional else 1), vocab_size
        )

    def forward(self, input_seq):
        """
        Parameters
        ----------
        seq : 
            batch_size x seq_len
        """
        batch_size, seq_len = input_seq.size()
        embedding = self.embedding(input_seq)
        _, hidden = self.encoder_rnn(embedding)

        # Take last hidden state
        if self.bidirectional or self.num_layers > 1:
            hidden = hidden.view(batch_size, self.hidden_dim * self.hidden_factor)
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
            hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_dim)
        else:
            hidden = hidden.unsqueeze(0)

        embedding = self.decoder_dropout(embedding)

        #  Decoder Forward
        output, _ = self.decoder_rnn(embedding, hidden)
        b, s, _ = output.size()
        # Output to Vocab
        logp = nn.functional.log_softmax(self.output_to_vocab(output), dim=-1)
        logp = logp.view(b, s, self.embedding.num_embeddings)
        return logp, mean, logv, z

    def infer(self, n=4, z=None):
        # TODO: Fix Bugs
        # Create hidden
        if z is None:
            batch_size = n
            z = torch.randn([batch_size, self.latent_dim]).to(self.device)
        else:
            batch_size = z.size(0)

        # Decode
        hidden = self.latent_to_hidden(z)
        if self.bidirectional or self.num_layers > 1:
            hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_dim)
        else:
            hidden = hidden.unsqueeze(0)

        sequence_id = torch.arange(0, batch_size).long().to(self.device)
        sequence_mask = torch.ones(batch_size).type(torch.bool).to(self.device)
        current_sequence = torch.arange(0, batch_size).long().to(self.device)
        remaining_sequence = torch.arange(0, batch_size).long().to(self.device)
        generated_sequences = (
            torch.Tensor(batch_size, self.seq_len).long().fill_(self.TEXT.vocab.stoi["<pad>"]).to(self.device)
        )

        t = 0
        while t < self.seq_len and len(remaining_sequence)>0:
            if t ==0 :
                input_seq = torch.Tensor(batch_size).long().fill_(self.TEXT.vocab.stoi["<sos>"]).to(self.device)

            if len(remaining_sequence) == 1 and len(input_seq.size()) == 0:
                input_seq = input_seq.unsqueeze(0)
            input_seq = input_seq.unsqueeze(-1)
            input_embedding = self.embedding(input_seq)

            output, hidden = self.decoder_rnn(input_embedding, hidden)
            logits = self.output_to_vocab(output)
            print(logits.size())
            _, sample = torch.topk(logits, 1, dim=-1)
            input_seq = sample.squeeze()
            print("Inut seq: ", input_seq.size())
            # Save the next word
            generated_sequences = self.save_sample(generated_sequences, input_seq, current_sequence, t)
            print("generated_sequences: ", generated_sequences.size())
            #print(input_seq != self.TEXT.vocab.stoi["<eos>"])
            print("EOS ID: ", self.TEXT.vocab.stoi['<eos>'])
            print("SOS ID: ", self.TEXT.vocab.stoi['<sos>'])
            print("Input Seq: ", input_seq)
            sequence_mask[current_sequence] = (input_seq != self.TEXT.vocab.stoi["<eos>"])
            print("seq mask: ", sequence_mask)
            current_sequence = sequence_id.masked_select(sequence_mask)
            print("curent seq:", current_sequence)
            remaining_mask = (input_seq != self.TEXT.vocab.stoi["<eos>"])
            print("remain mask: ", remaining_mask)
            remaining_sequence = remaining_sequence.masked_select(remaining_mask)
            print("remain seq: ", remaining_sequence)
            if len(remaining_sequence) > 0:
                if len(remaining_sequence) == 1 and len(input_seq.size()) == 0:
                        pass
                else:
                    input_seq = input_seq[remaining_sequence]
                    hidden = hidden[:, remaining_sequence]
                remaining_sequence = torch.arange(0, len(remaining_sequence)).long().to(self.device)
            t += 1

        return generated_sequences, z

    def save_sample(self, save_to, sample, current_sequence, t):
        current = save_to[current_sequence]
        current[:, t] = sample
        save_to[current_sequence] = current
        print("current save: ", current)
        return save_to
