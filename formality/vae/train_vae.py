import click
import time
from sequence_vae import VAE
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Dataset
import spacy
from torchtext import data, datasets
import numpy as np
from collections import defaultdict
import json
import sys
from nltk.tokenize import TweetTokenizer
from collections import Counter

spacy_en = spacy.load("en")


class PennTreeBankDataset(Dataset):
    """Penn Tree Bank Dataset."""

    def __init__(
        self,
        path,
        split,
        load_dataset=True,
        max_seq_len=60,
        min_freq=3,
        pad="<pad>",
        unk="<unk>",
        sos="<sos>",
        eos="<eos>",
    ):
        self.path = path
        self.split = split.lower()
        self.data_file = self.path + "/ptb." + self.split + ".txt"
        self.data_save = self.path + "/ptb." + self.split + ".json"
        self.max_seq_len = max_seq_len
        self.min_freq = min_freq

        self.stoi = dict()
        self.itos = dict()
        self.pad = pad
        self.unk = unk
        self.sos = sos
        self.eos = eos
        self.pad_id = None
        self.unk_id = None
        self.sos_id = None
        self.eos_id = None
        self.vocab_size = None
        self.vocab_save = path + "/ptb.vocab.json"

        if load_dataset:
            self._load_dataset()
        else:
            self._create_dataset()

    def __len__(self,):
        return len(self.data)

    def __getitem__(self, idx):
        idx = str(idx)
        return {
            "text": torch.tensor(self.data[idx]["text"]),
            "target": torch.tensor(self.data[idx]["target"]),
            "length": self.data[idx]["length"],
        }

    def _create_dataset(self):
        # TODO: Use Bucket Iterator so remove padding from here
        # TODO: Remove load dataset from here
        if self.split == "train":
            self._create_vocab()
        else:
            self._load_vocab()

        tokenizer = TweetTokenizer(preserve_case=False)
        data = defaultdict(dict)
        with open(self.data_file, "r") as fobj:
            for line in fobj:
                words = tokenizer.tokenize(line)
                text = [self.sos] + words
                text = text[: self.max_seq_len]
                target = words[: self.max_seq_len - 1]
                target.append(self.eos)
                length = len(text)
                text.extend([self.pad] * (self.max_seq_len - length))
                target.extend([self.pad] * (self.max_seq_len - length))

                text = [self.stoi.get(word, self.stoi["<unk>"]) for word in text]
                target = [self.stoi.get(word, self.stoi["<unk>"]) for word in target]
                idx = len(data)
                data[idx]["text"] = text
                data[idx]["target"] = target
                data[idx]["length"] = length

        with open(self.data_save, "wb") as fobj:
            data = json.dumps(data, ensure_ascii=False)
            fobj.write(data.encode("utf8", "replace"))

        self._load_dataset(vocab=False)

    def _load_dataset(self, vocab=True):
        with open(self.data_save, "rb") as fobj:
            self.data = json.load(fobj)
        if vocab:
            self._load_vocab()

    def _create_vocab(self,):
        # TODO: Use Spacy tokenizer and handle unk
        # TODO: Use Pickling
        tokenizer = TweetTokenizer(preserve_case=False)
        word_counter = Counter()
        sp_tok = [self.pad, self.unk, self.sos, self.eos]
        for tok in sp_tok:
            self.itos[len(self.stoi)] = tok
            self.stoi[tok] = len(self.stoi)
        self.pad_id = self.stoi[self.pad]
        self.unk_id = self.stoi[self.unk]
        self.sos_id = self.stoi[self.sos]
        self.eos_id = self.stoi[self.eos]

        with open(self.data_file, "r") as fobj:
            for line in fobj:
                words = tokenizer.tokenize(line)
                word_counter.update(words)

        for word, count in word_counter.items():
            if count > self.min_freq and word not in sp_tok:
                idx = len(self.stoi)
                self.itos[idx] = word
                self.stoi[word] = idx

        print("*" * 84)
        print("Vocabulary Creation Complete.")
        print("Number of words in vocab: {}".format(len(self.stoi)))
        self.vocab_size = len(self.stoi)
        vocab = {"stoi": self.stoi, "itos": self.itos}
        with open(self.vocab_save, "wb") as fobj:
            data = json.dumps(vocab, ensure_ascii=False)
            fobj.write(data.encode("utf8", "replace"))

    def _load_vocab(self):
        with open(self.vocab_save, "rb") as fobj:
            vocab = json.load(fobj)
        self.stoi = vocab["stoi"]
        self.itos = vocab["itos"]


def get_experiment_name(
    embedding_dim,
    hidden_dim,
    latent_dim,
    num_layers,
    max_seq_len,
    dropout,
    word_dropout,
    batch_size,
    rnn_type,
    bidirectional,
    learning_rate,
    epochs,
    ts,
):
    """Helper func to set log directory name."""
    name = "exp_ed{}_hd{}_ld{}_nl{}_seqlen{}_dp{}_wdp{}_bs{}_{}_lr{}_epochs{}_".format(
        embedding_dim,
        hidden_dim,
        latent_dim,
        num_layers,
        max_seq_len,
        dropout,
        word_dropout,
        batch_size,
        rnn_type,
        learning_rate,
        epochs,
    )
    if bidirectional:
        name += "bidirectional_"
    name += ts
    return name


def kld(logvar, mean, step, k, x0):
    kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    kl_weight = float(1 / (1 + np.exp(-k * (step - x0))))
    return kl_loss, kl_weight


def rev(batch, train_dataset):
    res = ""
    for i in range(batch.size(0)):
        res += str(i) + " : " + " ".join([train_dataset.itos[word.item()] for word in batch[i] if word.item() != train_dataset.stoi['<pad>']])
        res += "\n"
    return res


def tokenizer(text):  # create a tokenizer function
    text = text.replace("<unk>", "unk")
    return [tok.text for tok in spacy_en.tokenizer(text)]


@click.command()
@click.option(
    "-ed", "--embedding-dim", default=300, show_default=True, help="Embedding Dimension"
)
@click.option(
    "-hd", "--hidden-dim", default=256, show_default=True, help="Hidden Dimension"
)
@click.option(
    "-ld", "--latent-dim", default=16, show_default=True, help="Latent Dimension"
)
@click.option(
    "-nl", "--num-layers", default=1, show_default=True, help="Number of Layers"
)
@click.option(
    "-sl", "--max-seq-len", default=60, show_default=True, help="Max Sequence Length"
)
@click.option("-dp", "--dropout", default=0.5, show_default=True, help="Dropout")
@click.option(
    "-wd", "--word-dropout", default=0.0, show_default=True, help="Word Dropout"
)
@click.option("-bs", "--batch-size", default=32, show_default=True, help="Batch Size")
@click.option(
    "-rt",
    "--rnn-type",
    default="gru",
    show_default=True,
    help="RNN Type [gru, rnn, lstm]",
)
@click.option(
    "--bidirectional/--unidirectional",
    default=False,
    show_default=True,
    help="Bidirectional RNN",
)
@click.option(
    "-lr", "--learning-rate", default=0.001, show_default=True, help="Learning Rate"
)
@click.option("-ne", "--epochs", default=10, show_default=True, help="Number of Epochs")
@click.option(
    "-l",
    "--tensorboard-log",
    default="./logs/",
    show_default=True,
    help="Tensorboard Log Dir",
)
@click.option(
    "-i",
    "--input-data-dir",
    default="../data/",
    show_default=True,
    help="Input Data Directory",
)
@click.option(
    "-o",
    "--output-data-dir",
    default="../models/",
    show_default=True,
    help="Model Save Directory",
)
@click.option("-d", "--device", default="cpu", show_default=True, help="Device")
def main(
    embedding_dim,
    hidden_dim,
    latent_dim,
    num_layers,
    max_seq_len,
    dropout,
    word_dropout,
    batch_size,
    rnn_type,
    bidirectional,
    learning_rate,
    epochs,
    tensorboard_log,
    input_data_dir,
    output_data_dir,
    device,
):
    """Train VAE Model."""
    ts = time.strftime("%Y-%b-%d-%H-%M-%S", time.gmtime())

    # 1. Create Dataset
    train_dataset = PennTreeBankDataset(
        path=input_data_dir, split="train", load_dataset=False,
    )
    valid_dataset = PennTreeBankDataset(
        path=input_data_dir, split="valid", load_dataset=False,
    )
    test_dataset = PennTreeBankDataset(
        path=input_data_dir, split="test", load_dataset=False,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )
    # test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, pin_memory=torch.cuda.is_available())
    # TEXT = data.Field(
    #     init_token="<sos>",
    #     eos_token="<eos>",
    #     unk_token="unk",
    #     tokenize=tokenizer,
    #     batch_first=True,
    #     lower=True,
    #     #fix_length=max_seq_len,
    # )
    # train_dataset, val_dataset, test_dataset = datasets.PennTreebank.splits(
    #     text_field=TEXT, root=input_data_dir
    # )
    #
    # TEXT.build_vocab(train_dataset)
    # vocab_size = len(TEXT.vocab)
    # print("Vocab Size: {}".format(vocab_size))

    # # 2. Create Iterator
    # train_iterator, val_iterator, test_iterator = data.BucketIterator.splits(
    #     (train_dataset, val_dataset, test_dataset),
    #     batch_size=batch_size,
    #     sort_key = lambda x: len(x.text),
    #     sort_within_batch=False,
    #     device=device,
    # )
    # for i, b in enumerate(train_iterator):
    #     print(b.text.shape)
    #     sys.exit()
    # train_iterator, val_iterator, test_iterator = data.BPTTIterator.splits(
    #    (train_dataset, val_dataset, test_dataset),
    #    batch_size=batch_size,
    #    bptt_len=max_seq_len,
    #    device=device,
    # )

    # 3. Create Model
    model = VAE(
        train_dataset.vocab_size,
        embedding_dim,
        hidden_dim,
        latent_dim,
        max_seq_len,
        num_layers,
        rnn_type,
        device,
        dropout,
        word_dropout,
        bidirectional,
        pad_id=train_dataset.pad_id,
        unk_id=train_dataset.unk_id,
        sos_id=train_dataset.sos_id,
        eos_id=train_dataset.eos_id,
    )

    model.to(device)

    # 4. Create tensorboard summary
    exp_name = get_experiment_name(
        embedding_dim,
        hidden_dim,
        latent_dim,
        num_layers,
        max_seq_len,
        dropout,
        word_dropout,
        batch_size,
        rnn_type,
        bidirectional,
        learning_rate,
        epochs,
        ts,
    )
    writer = SummaryWriter(tensorboard_log + exp_name)

    loss_function = torch.nn.NLLLoss(
        size_average=False, ignore_index=train_dataset.pad_id
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    def train(step, device, epoch, save):
        """Training Loop."""
        model.train()
        elbo = 0
        for i, batch in enumerate(train_loader):
            batch_size = batch["text"].size(0)
            text = batch["text"].to(device)  # batch_size x seq_len
            target = batch["target"].to(device)  # batch_size x seq_len
            # print("Input Size: ", text.size())
            # print("Output Size: ", target.size())
            logp, mean, logvar, z = model(text, batch["length"])
            # print("Logp Size: ", logp.size())
            # print("Mean Size: ", mean.size())
            # print("logvar Size: ", logvar.size())

            # Calculate loss
            logp = logp.view(-1, logp.size(2))
            # print("Resize logp: ", logp.size())
            target = target[:, :torch.max(batch['length']).item()].contiguous().view(-1)
            # target = target.view(-1)
            # print("Resize targte: ", target.size())
            
            nll_loss = loss_function(logp, target)
            kl_loss, kl_weight = kld(
                logvar, mean, i + epoch * len(train_loader), 0.0025, 2500
            )
            loss = (nll_loss + kl_loss * kl_weight) / batch_size
            # print("NLL Loss: ", nll_loss.item())
            # print("KL Loss: ", kl_loss.item())
            # print("KL Weight", kl_weight)
            if i == 3:
                print("True Sentences: ")
                print(rev(batch["text"][0:2], train_dataset))
                gen, _ = model.infer(z=z[:2])
                print("Target Sentences: ")
                print(rev(batch["target"][0:2], train_dataset))
                print("Predicted Sentences: ")
                print(rev(gen, train_dataset))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            writer.add_scalar("train/ELBO", loss.item(), i + epoch * len(train_loader))
            writer.add_scalar(
                "train/nll_loss",
                nll_loss.item() / batch_size,
                i + epoch * len(train_loader),
            )
            writer.add_scalar(
                "train/kl_loss",
                kl_loss.item() / batch_size,
                i + epoch * len(train_loader),
            )
            writer.add_scalar(
                "train/kl_weight",
                kl_weight / batch_size,
                i + epoch * len(train_loader),
            )
            elbo += loss.item()
        writer.add_scalar("train-epoch/elbo", elbo / len(train_loader), epoch)
        """
        for i ,batch in enumerate(train_iterator):
            print("True Sentence: ", rev(batch.text, TEXT))
            _, _, _, z = model(batch.text)
            gen, _ = model.infer(z=z)
            print("Predicted Sentence: ", rev(gen, TEXT))
            break
        """
        if save:
            checkpoint = output_data_dir + exp_name + str(epoch) + ".pt"
            torch.save(model.state_dict(), checkpoint)
        return loss.item(), nll_loss.item(), kl_loss.item()

    def eval(step, device, epoch, latent_store, store=False):
        """Evaluation Loop."""
        model.eval()
        elbo = 0
        for i, batch in enumerate(valid_loader):
            batch_size = batch["text"].size(0)
            text = batch["text"].to(device)  # batch_size x seq_len
            target = batch["target"].to(device)  # batch_size x seq_len
            logp, mean, logvar, z = model(text, batch["length"])
            # Calculate loss
            logp = logp.view(-1, logp.size(2))
            target = target[:, :torch.max(batch['length']).item()].contiguous().view(-1)
            # target = target.view(-1)
            nll_loss = loss_function(logp, target)
            kl_loss, kl_weight = kld(
                logvar, mean, (epoch + 1) * len(valid_loader), 0.0025, 2500
            )
            loss = (nll_loss + kl_loss * kl_weight) / batch_size

            writer.add_scalar("val/ELBO", loss.item(), i + epoch * len(valid_loader))
            writer.add_scalar(
                "val/nll_loss",
                nll_loss.item() / batch_size,
                i + epoch * len(valid_loader),
            )
            writer.add_scalar(
                "val/kl_loss",
                kl_loss.item() / batch_size,
                i + epoch * len(valid_loader),
            )
            writer.add_scalar(
                "val/kl_weight", kl_weight / batch_size, i + epoch * len(valid_loader)
            )
            elbo += loss.item()
            store = False
            # if store:
            #     if "target" not in latent_store:
            #         latent_store["target"] = list()
            #     latent_store["target"] += TEXT.reverse(target.view(batch_size, -1))
            #     if "z" not in latent_store:
            #         latent_store["z"] = z
            #     else:
            #         latent_store["z"] = torch.cat((latent_store["z"], z), dim=0)
        writer.add_scalar("val-epoch/elbo", elbo / len(valid_loader), epoch)
        if store:
            latent_variables = {
                "target": latent_store["target"],
                "z": latent_store["z"].tolist(),
            }
            with open(output_data_dir + "dump_" + exp_name, "w") as f:
                json.dump(latent_variables, f)
        return loss.item(), nll_loss.item(), kl_loss.item()

    step = 0
    latent_store = {}
    for epoch in range(epochs):
        save = False
        if epoch == epochs - 1:
            save = True
        loss, n, k = train(step, device, epoch, save)
        print("Train Loss: ", loss, "\t", n, "\t", k)
        loss, n, k = eval(step, device, epoch, latent_store, save)
        print("Eval Loss: ", loss, "\t", n, "\t", k)

    # with open(output_data_dir + "/" + exp_name + "_field.pkl", 'wb') as fobj:
    #     torch.save(TEXT, fobj)


if __name__ == "__main__":
    main()
