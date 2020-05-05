"""Train a VAE Model."""

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

from dataset import PennTreeBankDataset

spacy_en = spacy.load("en")




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
    """Estimate KL Loss. """
    kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    kl_weight = float(1 / (1 + np.exp(-k * (step - x0))))
    return kl_loss, kl_weight


def rev(batch, train_dataset):
    """Reverse string from tokens to words."""
    res = ""
    for i in range(batch.size(0)):
        res += str(i) + " : " #+ " ".join([train_dataset.itos[word.item()] for word in batch[i] if word.item() != train_dataset.stoi['<pad>']])
        for word in batch[i]:
            if word.item() == train_dataset.eos_id:
                res += '.'
                break
            elif word.item() != train_dataset.pad_id and word.item() != train_dataset.sos_id:
                res += ' '
                res += train_dataset.itos[word.item()]
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
@click.option("-st", "--steep", default=0.0025, show_default=True, help="Steepnes of KL Annealing")
@click.option("-ct", "--centre", default=2500, show_default=True, help="Centre of KL Annealing function, KL Weight =0.5 at this iteration.")
@click.option("-sg", "--strategy", default='random', show_default=True, help="Strategy for decoding. Options: [random, greedy].")
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
    default="../data/penn-treebank/",
    show_default=True,
    help="Input Data Directory",
)
@click.option(
    "-o",
    "--output-data-dir",
    default="./models/",
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
    steep,
    centre,
    strategy,
    tensorboard_log,
    input_data_dir,
    output_data_dir,
    device,
):
    """Train VAE Model."""
    ts = time.strftime("%Y-%b-%d-%H-%M-%S", time.gmtime())

    # 1. Create Dataset
    train_dataset = PennTreeBankDataset(
        path=input_data_dir, split="train", load_dataset=True,
    )
    valid_dataset = PennTreeBankDataset(
        path=input_data_dir, split="valid", load_dataset=True,
    )
    test_dataset = PennTreeBankDataset(
        path=input_data_dir, split="test", load_dataset=True,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )

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

    if torch.cuda.device_count() > 1:
        nn.DataParallel(model)
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
            logp, mean, logvar, z = model(text, batch["length"])

            # Calculate loss
            logp = logp.view(-1, logp.size(2))
            target = target[:, :torch.max(batch['length']).item()].contiguous().view(-1)
            
            nll_loss = loss_function(logp, target)
            kl_loss, kl_weight = kld(
                logvar, mean, i + epoch * len(train_loader), steep, centre
            )
            loss = (nll_loss + kl_loss * kl_weight) / batch_size
            if i == len(train_loader)-1:
                print("True Sentences: ")
                print(rev(batch["text"][0:10], train_dataset))
                #gen, _ = model.infer(z=z[:10], strategy=strategy)
                gen = model.beam_search(z = z[:10], beam_width=5)
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
                kl_weight,
                i + epoch * len(train_loader),
            )
            elbo += loss.item()
        writer.add_scalar("train-epoch/elbo", elbo / len(train_loader), epoch)
        if save:
            checkpoint = output_data_dir + exp_name + str(epoch) + ".pt"
            torch.save(model.state_dict(), checkpoint)
        return loss.item()/batch_size, nll_loss.item()/batch_size, kl_loss.item()/batch_size

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
                logvar, mean, (epoch + 1) * len(valid_loader), steep, centre
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
                "val/kl_weight", kl_weight, i + epoch * len(valid_loader)
            )
            elbo += loss.item()
        writer.add_scalar("val-epoch/elbo", elbo / len(valid_loader), epoch)
        # if store:
        #     latent_variables = {
        #         "target": latent_store["target"],
        #         "z": latent_store["z"].tolist(),
        #     }
        #     with open(output_data_dir + "dump_" + exp_name, "w") as f:
        #         json.dump(latent_variables, f)
        return loss.item()/batch_size, nll_loss.item()/batch_size, kl_loss.item()/batch_size

    step = 0
    latent_store = {}
    for epoch in range(epochs):
        save = False
        if epoch == epochs - 1:
            save = True
        loss, n, k = train(step, device, epoch, save)
        print("*"*25 + 'Epoch : ' + str(epoch) + "*"*25)
        print("{:7}{:15}{}{:15}{}{:15}".format('', 'Total Loss', '\t', 'NLL Loss', '\t', 'KLL Loss'))
        print("{:7}{:.5f}{}{:.5f}{}{:.5f}".format('Train:', loss, '\t', n, '\t', k))
        loss, n, k = eval(step, device, epoch, latent_store, save)
        print("{:7}{:.5f}{}{:.5f}{}{:.5f}".format('Valid:', loss, '\t', n, '\t', k))



if __name__ == "__main__":
    main()
