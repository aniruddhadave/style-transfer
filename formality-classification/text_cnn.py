from torch.utils.data import Dataset, DataLoader
from torchtext import data
import spacy
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch


def read_file(path, sentences, scores):
    with open(path, "r") as f:
        for line in f:
            elems = line.strip().split("\t")
            scores.append(float(elems[0]))
            sentences.append(elems[3])


def load_dataset(file_path):
    pass


class FormalityDataset(Dataset):
    def __init__(self):
        with open(".data/formality-corpus/"):
            self.sentences = []
            self.tag = []
            read_file("./data/formality-corpus/answers", self.sentences, self.scores)
            read_file("./data/formality-corpus/blog", self.sentences, self.scores)
            read_file("./data/formality-corpus/email", self.sentences, self.scores)
            read_file("./data/formality-corpus/news", self.sentences, self.scores)


class TextCNN(nn.Module):
    def __init__(
        self, vocab_size, filter_sizes, num_filters, embedding_dim, dropout=0.2
    ):
        super(TextCNN, self).__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(self.vocab_size, embedding_dim)
        self.conv1 = nn.ModuleList(
            [
                nn.Conv2d(1, num_filters, kernel_size=(k, embedding_dim))
                for k in filter_sizes
            ]
        )
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(len(filter_sizes) * num_filters, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # batch_size x seq_len x embedding_dim
        x = self.embedding(x)
        x = x.unsqueeze(1)
        x = [conv(x).squeeze(3) for conv in self.conv1]
        x = [self.relu1(i) for i in x]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = self.fc1(x)
        x = self.sig(x)
        return x.squeeze()


def train_model(model, train_iterator, loss_function, optimizer):
    train_loss = 0
    correct = 0
    model.train()
    for bi, batch in enumerate(train_iterator):
        # 10
        label = batch.label
        # print(label.dtype)
        # 10 x 80
        text = batch.text
        optimizer.zero_grad()
        pred = model(text)
        # print("label: ", label.float())
        # print("pred: ", pred)
        loss = loss_function(pred, label.float())
        # print("loss: ", loss.item())
        train_loss += loss.item()
        pred = pred > 0.5
        correct += (pred == label).sum()
        # print("Correct: ", correct)
        # print(pred)
        # print(label)
        # print(correct)
        loss.backward()
        optimizer.step()
    return train_loss, correct


def eval_model(model, eval_iterator, loss_function):
    eval_loss = 0
    correct = 0
    model.eval()
    with torch.no_grad():
        for bi, batch in enumerate(eval_iterator):
            # 10
            label = batch.label
            # 10 x 80
            text = batch.text
            pred = model(text)
            loss = loss_function(pred, label.float())
            eval_loss += loss.item()
            correct += (label == pred).sum()

    return eval_loss, correct


def main():
    device = "cpu"
    BATCH_SIZE = 10
    learning_rate = 0.1
    num_epoch = 10

    TEXT = data.Field(fix_length=80, tokenize="spacy", batch_first=True, lower=True)
    LABEL = data.LabelField(batch_first=True, use_vocab=False)
    fields = {"label": ("label", LABEL), "sentence": ("text", TEXT)}

    train_data, valid_data, test_data = data.TabularDataset.splits(
        path="./data/formality-corpus/",
        train="train.json",
        validation="dev.json",
        test="test.json",
        format="json",
        fields=fields,
    )
    TEXT.build_vocab(train_data)
    train_iterator, valid_iterator, test_iterator = data.Iterator.splits(
        (train_data, valid_data, test_data),
        sort=False,  # don't sort test/validation data
        batch_size=BATCH_SIZE,
        device=device,
    )

    model = TextCNN(len(TEXT.vocab), [1], 64, 128)
    model.to(device)

    loss_function = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epoch):
        train_loss, train_acc = train_model(
            model, train_iterator, loss_function, optimizer
        )
        eval_loss, eval_acc = eval_model(model, valid_iterator, loss_function)
        print(
            "Epoch: {:2}, Train Loss: {:.5f}, Train Acc: {:.5f}%, Val. Loss:{:.5f}, Val. Acc:{:.5f}%".format(
                epoch,
                train_loss / len(train_data),
                float(train_acc) / len(train_data),
                eval_loss / len(valid_data),
                float(eval_acc) / len(valid_data),
            )
        )


if __name__ == "__main__":
    main()
