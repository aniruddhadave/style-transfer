import torch
from torch.utils.data import DataLoader, Dataset
import spacy
import numpy as np
from collections import defaultdict
import json
import sys
from nltk.tokenize import TweetTokenizer
from collections import Counter
import pickle

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
        self.data_save = self.path + "/ptb." + self.split + ".pkl"
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
        self.vocab_save = path + "/ptb.vocab.pkl"
        self.data = None
        
        if load_dataset:
            self._load_dataset()
        else:
            self._create_dataset()

    def __len__(self,):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "text": torch.tensor(self.data[idx]["text"]),
            "target": torch.tensor(self.data[idx]["target"]),
            "length": self.data[idx]["length"],
        }

    def _create_dataset(self):
        # TODO: Use Bucket Iterator so remove padding from here
        # TODO: Remove load dataset from here
        print("Creating Dataset for {} split.".format(self.split))
        if self.split == "train":
            self._create_vocab()
        else:
            self._load_vocab()

        tokenizer = TweetTokenizer(preserve_case=False)
        self.data = defaultdict(dict)
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
                idx = len(self.data)
                self.data[idx]["text"] = text
                self.data[idx]["target"] = target
                self.data[idx]["length"] = length

        # with open(self.data_save, "wb") as fobj:
        #     data = json.dumps(data, ensure_ascii=False)
        #     fobj.write(data.encode("utf8", "replace"))
        with open(self.data_save, 'wb') as fobj:
            pickle.dump(self.data, fobj, protocol=pickle.HIGHEST_PROTOCOL)
        self._load_dataset(vocab=False)

    def _load_dataset(self, vocab=True):
        # with open(self.data_save, "rb") as fobj:
        #     self.data = json.load(fobj)
        print("Loading dataset for {} split.".format(self.split))
        with open(self.data_save, 'rb') as fobj:
            self.data = pickle.load(fobj)
        if vocab:
            self._load_vocab()

    def _create_vocab(self,):
        # TODO: Use Spacy tokenizer and handle unk
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
            # data = json.dumps(vocab, ensure_ascii=False)
            # fobj.write(data.encode("utf8", "replace"))
            pickle.dump(vocab, fobj, protocol=pickle.HIGHEST_PROTOCOL)

    def _load_vocab(self):
        print("Loading Vocabulary.")
        with open(self.vocab_save, "rb") as fobj:
            # vocab = json.load(fobj)
            vocab = pickle.load(fobj)

        self.stoi = vocab["stoi"]
        self.itos = vocab["itos"]
        self.vocab_size = len(self.stoi)
        self.pad_id = self.stoi[self.pad]
        self.unk_id = self.stoi[self.unk]
        self.sos_id = self.stoi[self.sos]
        self.eos_id = self.stoi[self.eos]
