#!/usr/bin/env python
import os
import string
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import random
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import nltk
import torch.utils.data as Data
nltk.download('punkt')
from nltk import word_tokenize
import pickle
# from sklearn.model_selection import train_test_split
# import time

from Trie import Trie
from Trie import append_to_dict
from MultiLM import MultiLM
import tqdm
"""
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
    vocab = list(token_set)
    word2idx = {w: idx+1 for (idx, w) in enumerate(vocab)}
    word2idx['<pad>'] = 0
    idx2word = {idx+1 : w for (idx, w) in enumerate(vocab)}
    idx2word[0] = '<pad>'
"""

def one_hot(k): 
    v = np.zeros(len(word2idx))
    v[k] = 1
    return v
    
def data_helper(t_corpus):
    n_rows = sum([len(sen) for sen in t_corpus]) - len(t_corpus)
    print('n_rows:', n_rows)
    vectorized_sents = [[word2idx[tok] for tok in sent if tok in word2idx] for sent in t_corpus]
    sent_tensor = torch.zeros((n_rows, 4)).int()
    labels = []
    k = 0

    print("vectorized_sents", len(vectorized_sents))
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
                sent_tensor[k][0] = sen[i-3]
                sent_tensor[k][1] = sen[i-2]
                sent_tensor[k][2] = sen[i-1]
                sent_tensor[k][3] = sen[i]
            labels.append(sen[i+1])
            k += 1
    
    labels = torch.tensor(labels)
    print('label shape', labels.shape)
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
    for sent_tensor, labels in enumerate(loader):
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

def run_pred(data, word2idx, idx2word, trie):
    # start_time = time.time()
    # print(start_time)
    preds = []
    all_chars = string.ascii_letters
    for inp in tqdm.tqdm(data):
        curr = inp.split(' ')[-1]
        old = inp.split(' ')[:-1]

        # if the last one word is " ", we use rnn
        if curr == '':
            test_tensor = torch.zeros((1, 4)).int().to(device)

            for i in range(4):
                if len(old) >= i + 1 and old[i] in word2idx.keys():
                    test_tensor[0][i] = word2idx[old[i]]

            hidden = torch.zeros(1, 1, n_hidden).to(device)
            out = torch.nn.functional.softmax(model(test_tensor, hidden), dim=1)
        
            # rnn_time = time.time()
            # print('rnn_time', rnn_time - start_time)
            indices_list = torch.topk(out[0], 3).indices
            best_chars = []
            for indices in indices_list:
              best_chars.append(idx2word[indices.item()][0])

            preds.append(''.join(best_chars))
        else:
            res = trie.advance_curr(curr)
            words = []
            if res == 1:
                words = trie.get_next_char()
            top_guesses = []
            for word in words:
                top_guesses.append(word)
            for _ in range(3 - len(top_guesses)):
                top_guesses.append(random.choice(all_chars))
            preds.append(''.join(top_guesses))
            # trie_time = time.time()
            # print('trie time:', trie_time - start_time)
            # start_time = time.time()
        
    return preds


# def load_word2idx():
#     with open('src/saved_dictionary.pkl', 'rb') as f:
#         loaded_dict = pickle.load(f)
#     return loaded_dict
    
    
def load_vocabtrie(path):
     # TODO: filename has to be saved_dictionary_vocab.pkl
     with open(path, 'rb') as f:
         count = pickle.load(f)
     return count


def get_vocab():
    df = pd.read_csv("src/Language Detection.csv")
    corpus = df['Text'].values.tolist()
    # X_train, X_test = train_test_split(corpus, test_size=0.2, random_state=42)

    count = {}
    vocab = []
    # token_set = set()
    train_tokenized_corpus = []
    val_tokenized_corpus = []
    i = 0

    # for sentence in X_test:
    #     if i % 10000 == 0: print(i)
    #     tok_sen = word_tokenize(sentence)
    #     val_tokenized_corpus.append(tok_sen)
    #     i += 1

    # for sentence in X_train:
    #     if i % 10000 == 0: print(i)
    #     tok_sen = word_tokenize(sentence)
    #     train_tokenized_corpus.append(tok_sen)
    #     i += 1

    for sentence in corpus:
        if i % 10000 == 0: print(i)
        tok_sen = word_tokenize(sentence)
        for w in tok_sen:
            if w not in count:
                count[w] = 0
            count[w] += 1
        i += 1
    vocab = list(count.keys())
    word2idx = {w: idx+1 for (idx, w) in enumerate(vocab)}
    word2idx['<pad>'] = 0
    idx2word = {idx+1 : w for (idx, w) in enumerate(vocab)}
    idx2word[0] = '<pad>'
    vocab_trie = Trie(count)

    return word2idx, idx2word, vocab_trie, train_tokenized_corpus, val_tokenized_corpus


if __name__ == '__main__':
    n_hidden = 256
    SEQUENCE_LENGTH = 100
    EPOCHS = 30
    SEQUENCE_LENGTH = 100
    FEATURE_SIZE = 1024
    TEST_BATCH_SIZE = 256
    LEARNING_RATE = 0.002
    WEIGHT_DECAY = 0.0005
    PRINT_INTERVAL = 10
    BATCH_SIZE = 128

    device = torch.device("cuda")

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    args = parser.parse_args()

    file_path = args.work_dir

    if args.mode == 'train':
        if not os.path.isdir(args.work_dir):
            print('Making working directory {}'.format(args.work_dir))
            os.makedirs(args.work_dir)

        print('Instatiating model')
        device = torch.device("cuda")
        word2idx, idx2word, vocab_trie, train_tokenized_corpus, val_tokenized_corpus = get_vocab()
        feature_size = FEATURE_SIZE
        vocab_size = len(word2idx)
        model = MultiLM(vocab_size, feature_size).to(device)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        
        print('Loading training data')
        train_sent_tensor, train_labels = data_helper(train_tokenized_corpus)
        train_dataset = Data.TensorDataset(train_sent_tensor, train_labels)
        train_loader = Data.DataLoader(train_dataset, BATCH_SIZE, True) 

        print('Loading validation data')
        val_sent_tensor, val_labels = data_helper(val_tokenized_corpus)
        val_dataset = Data.TensorDataset(val_sent_tensor, val_labels)
        val_loader = Data.DataLoader(val_dataset, BATCH_SIZE, True) 

        print('Training')
        for epoch in range(EPOCHS):
            print("Epoch :", epoch)
            avg_loss = run_train(model, device, optimizer, train_loader, val_loader, LEARNING_RATE, epoch, PRINT_INTERVAL)
            if epoch % 5 == 0:
                print('Saving model')
                model.save_model('model.checkpoint')

    elif args.mode == 'test':
        device = torch.device("cuda")
        print('Loading vocab')
        # word2idx, idx2word, vocab_trie, train_tokenized_corpus, val_tokenized_corpus = get_vocab()
        count = load_vocabtrie('src/saved_count.pkl')
        vocab = list(count.keys())
        word2idx = {w: idx+1 for (idx, w) in enumerate(vocab)}
        word2idx['<pad>'] = 0
        idx2word = {idx+1 : w for (idx, w) in enumerate(vocab)}
        idx2word[0] = '<pad>'
        vocab_trie = Trie(count)

        print('Loading model')
        model = MultiLM(len(word2idx), FEATURE_SIZE).to(device)
        model.load_state_dict(torch.load('src/model.checkpoint', map_location=torch.device('cuda')))
        test_data = load_test_data(args.test_data)
        
        print('Making predictions')
        pred = run_pred(test_data, word2idx, idx2word, vocab_trie)
        
        print('Writing predictions to {}'.format(args.test_output))
        
        assert len(pred) == len(test_data), 'Expected {} predictions but got {}'.format(len(test_data), len(pred))
        write_pred(pred, args.test_output)
    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))
