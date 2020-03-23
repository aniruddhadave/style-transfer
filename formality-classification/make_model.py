from torch.utils.data import Dataset, DataLoader
import pickle
from torchtext import data
from torchtext import datasets
import torch
from torch import nn
import sys

class simple_classifier(nn.Module):
    def __init__(self, embedding_dim, vocab_size, hidden_dim, dropout=0, ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

        nn.init.normal_(self.embed.weight)

    def forward(self, x):
        x = self.embed(x)
        out, hidden = self.rnn(x)
        x = self.out(hidden.squeeze(0))
        return x

def main():

    BATCH_SIZE = 10
    learning_rate = 0.01
    num_epochs = 10

    SENTENCE = data.Field(init_token='</s>', eos_token='</e>', lower=False, tokenize="spacy", batch_first= False)
    SCORE = data.Field(sequential=False, is_target=True)
    fields = {'sentence':('text', SENTENCE), 'scores':('score', SCORE)}

    train_data, valid_data, test_data = data.TabularDataset.splits(path='./data', train='train.json', validation='dev.json', test='test.json', format='json', fields=fields)
    SENTENCE.build_vocab(train_data)
    SCORE.build_vocab(train_data)
    print(train_data[0])
    print(SENTENCE.vocab.freqs.most_common(10))

    device = 'cpu'
    train_itr, valid_itr, test_itr = data.BucketIterator.splits((train_data, valid_data, test_data), batch_size = BATCH_SIZE, device = device, sort=False)
    batch = next(train_itr.__iter__())
    print(batch)

    model  = simple_classifier(64, len(SENTENCE.vocab), 128, 0.2)
    optimizer = torch.optim.Adam(lr=learning_rate, params = model.parameters())
    loss_function = nn.BCEWithLogitsLoss()

    for epoch in range(num_epochs):
        model.train()
        per_epoch_loss = 0
        for i, batch in enumerate(train_itr):
            optimizer.zero_grad()
            text = batch.text
            label = batch.score - 1 
            out = model(text)
            loss = loss_function(out, label.unsqueeze(1).double())
            per_epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
        print("Train loss at Epoch ", epoch, " : ", per_epoch_loss)
        with torch.no_grad():
            model.eval()
            per_epoch_valid_loss = 0
            for i, batch in enumerate(valid_itr):
                text = batch.text
                label = batch.score -1
                out = model(text)
                per_epoch_valid_loss += loss_function(out, label.unsqueeze(1).double()).item()
            print("Valid Loss at Epoch ", epoch, " : ", per_epoch_valid_loss)

    batch = next(valid_itr.__iter__())
    text = batch.text[0]
    label = batch.score[0]-1
    print("-"*84)
    print("Text : " , text)
    print("Label: ", label)
    print("Prediction: ", model(text))


if __name__ == "__main__":
    main()
