import torch
import torch.nn as nn

class VAE(nn.Module):
    """Sequence level VAE."""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, latent_dim, max_seq_len, num_layers, rnn_type, device, embedding_dropout, word_dropout, bidirectional=False):
        super(VAE, self).__init__()

        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.device = device
        self.word_dropout = word_dropout
        self.hidden_dim = hidden_dim
        # Set rnn type
        if rnn_type == 'rnn':
            rnn = nn.RNN
        elif rnn_type == 'gru':
            rnn = nn.GRU
        elif rnn_type == 'lstm':
            rnn = nn.LSTM
        else:
            raise ValueError("Check rnn type.")

        self.hidden_factor = (2 if bidirectional else 1) * self.num_layers

        #Input
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        self.encoder_rnn = rnn(embedding_dim, hidden_dim, num_layers, batch_first=True, bidirectional=self.bidirectional)
        
        # Latent
        self.hidden_to_mean = nn.Linear(hidden_dim * self.hidden_factor, latent_dim)
        self.hidden_to_logv = nn.Linear(hidden_dim * self.hidden_factor, latent_dim)
        
        self.decoder_rnn = rnn(embedding_dim, hidden_dim, num_layers, batch_first=True, bidirectional=self.bidirectional)
        self.decoder_dropout = nn.Dropout(word_dropout)
        # Create intermediate and out linear layers
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim * self.hidden_factor)
        self.output_to_vocab = nn.Linear(hidden_dim * (2 if bidirectional else 1), vocab_size)


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
            hidden = hidden.view(batch_size, self.hidden_dim*self.hidden_factor)
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
        logp = nn.functional.log_softmax(self.output_to_vocab(output),dim=-1)
        logp = logp.view(b, s, self.embedding.num_embeddings)
        return logp, mean, logv, z

