import click
import time
from sequence_vae import VAE
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import spacy
from torchtext import data, datasets
import numpy as np
from collections import defaultdict
import json


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

    ts = time.strftime("%Y-%b-%d-%H-%M-%S", time.gmtime())

    # 1. Create Dataset and iterator
    TEXT = data.ReversibleField(
        init_token="<sos>",
        eos_token="<eos>",
        tokenize="spacy",
        batch_first=True,
        fix_length=max_seq_len,
    )
    train_dataset, val_dataset, test_dataset = datasets.PennTreebank.splits(
        text_field=TEXT, root=input_data_dir
    )
    TEXT.build_vocab(train_dataset)
    vocab_size = len(TEXT.vocab)
    train_iterator, val_iterator, test_iterator = data.BPTTIterator.splits(
        (train_dataset, val_dataset, test_dataset),
        batch_size=batch_size,
        bptt_len=max_seq_len,
        device=device,
    )

    # 2. Create Model
    model = VAE(
        vocab_size,
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
        TEXT
    )

    model.to(device)

    # Create tensorboard summary
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

    loss_function = torch.nn.NLLLoss()

    def kld(logvar, mean, step, k, x0):
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        kl_weight = float(1 / (1 + np.exp(-k * (step - x0))))
        return kl_loss, kl_weight

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    def train(step, device, epoch, save):
        model.train()
        elbo = 0
        for i, batch in enumerate(train_iterator):
            batch_size = batch.batch_size
            text = batch.text.to(device)
            target = batch.target.to(device)

            logp, mean, logvar, z = model(text)
            # Calculate loss
            logp = logp.view(-1, logp.size(2))
            target = target.view(-1)
            nll_loss = loss_function(logp, target)
            kl_loss, kl_weight = kld(
                logvar, mean, i + epoch * len(train_iterator), 0.0025, 2500
            )
            loss = (nll_loss + kl_loss * kl_weight) / batch_size

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            writer.add_scalar(
                "train/ELBO", loss.item(), i + epoch * len(train_iterator)
            )
            writer.add_scalar(
                "train/nll_loss",
                nll_loss.item() / batch_size,
                i + epoch * len(train_iterator),
            )
            writer.add_scalar(
                "train/kl_loss",
                kl_loss.item() / batch_size,
                i + epoch * len(train_iterator),
            )
            writer.add_scalar(
                "train/kl_weight",
                kl_weight / batch_size,
                i + epoch * len(train_iterator),
            )
            elbo += loss.item()
        writer.add_scalar("train-epoch/elbo", elbo / len(train_iterator), epoch)
        if save:
            checkpoint = output_data_dir + exp_name + str(epoch) + ".pt"
            torch.save(model.state_dict(), checkpoint)
        return loss.item(), nll_loss.item(), kl_loss.item()

    def eval(step, device, epoch, latent_store, store=False):
        model.eval()
        elbo = 0
        for i, batch in enumerate(val_iterator):
            batch_size = batch.batch_size
            text = batch.text.to(device)
            target = batch.target.to(device)
            logp, mean, logvar, z = model(text)
            # Calculate loss
            logp = logp.view(-1, logp.size(2))
            target = target.view(-1)
            nll_loss = loss_function(logp, target)
            kl_loss, kl_weight = kld(
                logvar, mean, (epoch + 1) * len(val_iterator), 0.0025, 2500
            )
            loss = (nll_loss + kl_loss * kl_weight) / batch_size

            writer.add_scalar("val/ELBO", loss.item(), i + epoch * len(val_iterator))
            writer.add_scalar(
                "val/nll_loss",
                nll_loss.item() / batch_size,
                i + epoch * len(val_iterator),
            )
            writer.add_scalar(
                "val/kl_loss",
                kl_loss.item() / batch_size,
                i + epoch * len(val_iterator),
            )
            writer.add_scalar(
                "val/kl_weight", kl_weight / batch_size, i + epoch * len(val_iterator)
            )
            elbo += loss.item()
            if store:
                if "target" not in latent_store:
                    latent_store["target"] = list()
                latent_store["target"] += TEXT.reverse(target.view(batch_size, -1))
                if "z" not in latent_store:
                    latent_store["z"] = z
                else:
                    latent_store["z"] = torch.cat((latent_store["z"], z), dim=0)
        writer.add_scalar("val-epoch/elbo", elbo / len(train_iterator), epoch)
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


    with open(output_data_dir + "/" + exp_name + "_field.pkl", 'wb') as fobj:
        torch.save(TEXT, fobj)

if __name__ == "__main__":
    main()
