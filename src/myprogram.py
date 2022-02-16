#!/usr/bin/env python
#！pip install datasets
import os
import string
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from datasets import load_dataset
dataset = load_dataset("amazon_reviews_multi", "all_languages")

import random
import numpy as np
import torch
import torch.optim as optim
import nltk
import torch.utils.data as Data
nltk.download('punkt')
from nltk import word_tokenize

import Trie
import MultiLM

n_hidden = 256
SEQUENCE_LENGTH = 100
BATCH_SIZE = 256
FEATURE_SIZE = 512
TEST_BATCH_SIZE = 256
EPOCHS = 20
LEARNING_RATE = 0.003
WEIGHT_DECAY = 0.0005
PRINT_INTERVAL = 10

word2idx = []
tokenized_corpus = []
vocab_size = 0

def load_training_data():
    train_df = dataset['train']
    val_df = dataset['validation']
    test_df = dataset['test']
    corpus = train_df["review_body"]

    vocab = []
    token_set = set()
    tokenized_corpus = []
    i = 0
    for sentence in corpus:
        if i % 10000 == 0: print(i)
        tok_sen = word_tokenize(sentence.lower())
        tokenized_corpus.append(tok_sen)
        token_set.update(tok_sen)
        i += 1
    #vocab = np.hstack(tokenized_corpus)
    #vocab = np.unique(vocab)
    vocab = list(token_set)
    word2idx = {w: idx+1 for (idx, w) in enumerate(vocab)}
    word2idx['<pad>'] = 0
    idx2word = {idx+1 : w for (idx, w) in enumerate(vocab)}
    idx2word[0] = '<pad>'

def one_hot(k): 
    v = np.zeros(len(word2idx))
    v[k] = 1
    return v
    
def data_helper():
    n_rows = sum([len(sen) for sen in tokenized_corpus]) - len(tokenized_corpus)
    vectorized_sents = [[word2idx[tok] for tok in sent if tok in word2idx] for sent in tokenized_corpus]
    sent_tensor = torch.zeros((n_rows, 4)).int()
    labels = []
    k = 0

    for sen in vectorized_sents:
        for i in range(len(sen)-1):
            if i == 0:
                sent_tensor[k][0] = sen[i]
            elif i == 1:
                sent_tensor[k][0] = sen[i-1]
                sent_tensor[k][1] = sen[i]
            elif i == 2:
                sent_tensor[k][0] = sen[i-2]
                sent_tensor[k][1] = sen[i-1]
                sent_tensor[k][2] = sen[i]
            else:
                sent_tensor[k][0] = sen[i-2]
                sent_tensor[k][1] = sen[i-2]
                sent_tensor[k][2] = sen[i-1]
                sent_tensor[k][3] = sen[i]
            labels.append(sen[i+1])
            k += 1
    labels = torch.tensor(labels)

    return sent_tensor, labels

def load_test_data(fname):
    # your code here
    data = []
    with open(fname) as f:
        for line in f:
            inp = line[:-1]  # the last character is a newline
            data.append(inp)
    return data

def write_pred(preds, fname):
    with open(fname, 'wt') as f:
        for p in preds:
            f.write('{}\n'.format(p))

def run_train(model, device, optimizer, loader, lr, epoch, log_interval):
    model.train()
    losses = []
    for batch_idx, (sent_tensor, labels) in enumerate(loader):
        data, label = sent_tensor.to(device), labels.to(device)
        hidden = torch.zeros(1, data.shape[0], n_hidden).to(device)
        pred = model(data, hidden)
        optimizer.zero_grad()
        loss = model.loss()
        loss = loss(pred, label)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    avg_loss = sum(losses) / len(losses)
    print('Average Loss', avg_loss)
    return avg_loss

def run_pred(data):
    # your code here
    preds = []
    all_chars = string.ascii_letters
    for inp in data:
        # this model just predicts a random character each time
        top_guesses = [random.choice(all_chars) for _ in range(3)]
        preds.append(''.join(top_guesses))
    return preds

'''
def save(work_dir):
    # your code here
    # this particular model has nothing to save, but for demonstration purposes we will save a blank file
    with open(os.path.join(work_dir, 'model.checkpoint'), 'wt') as f:
        f.write('dummy save')

def load(work_dir):
    # your code here
    # this particular model has nothing to load, but for demonstration purposes we will load a blank file
    with open(os.path.join(work_dir, 'model.checkpoint')) as f:
        dummy_save = f.read()
    return MyModel()
'''


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    args = parser.parse_args()

    random.seed(0)

    file_path = args.work_dir

    if args.mode == 'train':
        if not os.path.isdir(args.work_dir):
            print('Making working directory {}'.format(args.work_dir))
            os.makedirs(args.work_dir)
        print('Instatiating model')
        device = torch.device("cuda")
        feature_size = FEATURE_SIZE
        vocab_size = len(word2idx)
        model = MultiLM(vocab_size, feature_size).to(device)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        print('Loading training data')
        sent_tensor, labels = data_helper()
        dataset = Data.TensorDataset(sent_tensor, labels)
        loader = Data.DataLoader(dataset, BATCH_SIZE, True)        
        print('Training')
        for epoch in range(EPOCHS):
            print("Epoch :", epoch)
            avg_loss = run_train(model, device, optimizer, loader, LEARNING_RATE, epoch, PRINT_INTERVAL)
            print('Saving model')
            model.save_model(args.work_dir + 'model.checkpoint')
    elif args.mode == 'test':
        print('Loading model')
        model = MultiLM.load_model(args.work_dir)
        print('Loading test data from {}'.format(args.test_data))
        test_data = MultiLM.load_test_data(args.test_data)
        print('Making predictions')
        pred = model.run_pred(test_data)
        print('Writing predictions to {}'.format(args.test_output))
        assert len(pred) == len(test_data), 'Expected {} predictions but got {}'.format(len(test_data), len(pred))
        model.write_pred(pred, args.test_output)
    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))
