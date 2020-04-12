from torch.utils.data import Dataset, DataLoader
from torchtext import data, datasets
import spacy
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch
import click
import sys
import torchtext
import random

def read_file(path, sentences, scores):
    with open(path, "r") as f:
        for line in f:
            elems = line.strip().split("\t")
            scores.append(float(elems[0]))
            sentences.append(elems[3])


class FormalityDataset(Dataset):
    def __init__(self):
        with open(".data/formality-corpus/"):
            self.sentences = []
            self.tag = []
            read_file("./data/formality-corpus/answers", self.sentences, self.scores)
            read_file("./data/formality-corpus/blog", self.sentences, self.scores)
            read_file("./data/formality-corpus/email", self.sentences, self.scores)
            read_file("./data/formality-corpus/news", self.sentences, self.scores)

def generate_batch(batch):
    label = torch.tensor([entry[0] for entry in batch])
    text = [entry[1] for entry in batch]
    return text, label

class TextCNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        filter_sizes,
        num_filters,
        embedding_dim,
        num_classes,
        dropout=0.2,
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
        self.relu1 = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(len(filter_sizes) * num_filters, num_classes)

    def forward(self, x):
        x = self.embedding(x)  # [batch_size x seq_len x embed_dim]
        x = x.unsqueeze(1)  # [batch_size x in_filters x sseq_len x embed_dim]
        x = [
            conv(x).squeeze(3) for conv in self.conv1
        ]  # [batch_size x out_filters x seq_len] * k
        x = [self.relu1(i) for i in x]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]

        x = torch.cat(x, 1)  # [batch_size x out_filer * k]
        logits = self.fc1(x)  # [batch_size x num_classes]

        # Predict
        probs = F.log_softmax(logits, dim=1)
        classes = torch.max(probs, 1)[1]  # [1] for indices
        # x = F.softmax(x)
        return logits, classes


def train_model(model, train_iterator, loss_function, optimizer, device):
    train_loss = 0
    correct = 0
    model.train()
    for bi, batch in enumerate(train_iterator):
        # 10
        label = batch.label.to(device)
        # 10 x 80
        text = batch.text.to(device)
        optimizer.zero_grad()
        pred, classes = model(text)
        loss = loss_function(pred, label.long())
        train_loss += loss.item()
        correct += (classes == label).sum()
        loss.backward()
        optimizer.step()
    return train_loss, correct


def eval_model(model, eval_iterator, loss_function, device):
    eval_loss = 0
    correct = 0
    model.eval()
    with torch.no_grad():
        for bi, batch in enumerate(eval_iterator):
            # 10
            label = batch.label.to(device)
            # 10 x 80
            text = batch.text.to(device)
            pred, classes = model(text)
            loss = loss_function(pred, label.long())
            eval_loss += loss.item()
            correct += (label == classes).sum()

    return eval_loss, correct


@click.command()
@click.option("-d", "--device", default="cpu", show_default=True, help="Device Type")
@click.option("-ds", "--dataset", default="enron", show_default=True, help="Dataset. Currently Supported: [enron, AmazonReviewPolarity]")
@click.option(
    "-ml", "--max-len", default=80, show_default=True, help="Max Length of Sequences"
)
@click.option("-b", "--batch-size", default=64, show_default=True, help="Batch Size")
@click.option(
    "-lr", "--learning-rate", default=0.0005, show_default=True, help="Learning Rate"
)
@click.option(
    "-ne", "--num-epochs", default=4, show_default=True, help="Number of Epochs"
)
@click.option(
    "-c", "--num-classes", default=2, show_default=True, help="Number of Classes"
)
@click.option("-dp", "--dropout", default=0.5, show_default=True, help="Dropout")
@click.option(
    "-ed", "--embedding-dim", default=128, show_default=True, help="Embedding Dim"
)
@click.option(
    "-nf", "--num-filters", default=128, show_default=True, help="Number of Filters"
)
@click.option(
    "-fs",
    "--filter-sizes",
    default="1,2,3,4,5",
    show_default=True,
    help="Size of Filters to apply"
)
def main(
    device,
    dataset,
    max_len,
    batch_size,
    learning_rate,
    num_epochs,
    num_classes,
    dropout,
    embedding_dim,
    num_filters,
    filter_sizes,
):
    """Text CNN."""
    SEED = 1234
    print("Running on Device: ", device)
    if dataset == "enron":
        TEXT = data.Field(
            fix_length=max_len, tokenize="spacy", batch_first=True, lower=True
        )
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
            batch_size=batch_size,
            device=device,
        )
        vocab_size = len(TEXT.vocab)
    elif dataset == "AmazonReviewPolarity":
        print("Reading amazon dataset")
        TEXT = data.Field(
            lower=True, batch_first=True, fix_length=max_len, tokenize="spacy"
        )
        LABEL = data.Field(sequential=False)
        fields = [("label", LABEL), ("name", None), ("text", TEXT)]
        train_data, valid_data = data.TabularDataset.splits(
            path="./data/amazon_review_polarity_csv/",
            train="train.csv",
            test="test.csv",
            format="CSV",
            fields=fields,
        )
        TEXT.build_vocab(train_data)
        LABEL.build_vocab(train_data)
        train_iterator, valid_iterator = data.BucketIterator.splits(
            (train_data, valid_data),
            batch_size=batch_size,
            sort_key=lambda x: x.text,
            sort_within_batch=False,
            device=device,
        )
    elif dataset == "Yelp":
        print("Reading Yelp Dataset")
        train_data, test_data = datasets.YelpReviewPolarity(root='./data/', ngrams=1)
        train_iterator =  DataLoader(train_data, batch_size = batch_size, shuffle=True, collate_fn=generate_batch)
        valid_iterator =  DataLoader(test_data, batch_size = batch_size, shuffle=True, collate_fn=generate_batch)
        vocab_size = len(train_data.get_vocab())
    elif dataset == "IMDB":
        print("Reading IMDB dataset")
        TEXT = data.Field(tokenize = 'spacy', batch_first = True, lower=True)
        LABEL = data.LabelField(dtype = torch.float)
        train_data, test_data = datasets.IMDB.splits(TEXT, LABEL, root='./data/')
        train_data, valid_data = train_data.split(random_state=random.seed(SEED))
        TEXT.build_vocab(train_data)
        LABEL.build_vocab(train_data)
        train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
                (
                    train_data, valid_data, test_data
                    ),
                batch_size = batch_size, 
                device = device
                )
        vocab_size = len(TEXT.vocab)
    else:
        raise ValueError("Dataset: {} currently not supported".format(dataset))
    
    print("Reading dataset completed.")
    filter_sizes = [int(x) for x in filter_sizes.split(",")]
    model = TextCNN(
        vocab_size,
        [1, 2, 3, 4, 5],
        num_filters,
        embedding_dim,
        num_classes,
        dropout,
    )
    model.to(device)
    print("Model Initialized")
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        print("Epoch:", epoch)
        train_loss, train_acc = train_model(
            model, train_iterator, loss_function, optimizer, device
        )
        eval_loss, eval_acc = eval_model(model, valid_iterator, loss_function, device)
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
