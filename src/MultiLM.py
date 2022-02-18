import torch
import torch.nn as nn

n_hidden = 256
SEQUENCE_LENGTH = 100
BATCH_SIZE = 256
FEATURE_SIZE = 512
TEST_BATCH_SIZE = 256
EPOCHS = 20
LEARNING_RATE = 0.003
WEIGHT_DECAY = 0.0005
PRINT_INTERVAL = 10
batch_size = 256

class MultiLM(nn.Module):
    def __init__(self, vocab_size, feature_size):
        super(MultiLM, self).__init__()
        self.vocab_size = vocab_size
        self.feature_size = feature_size
        self.encoder = nn.Embedding(self.vocab_size, self.feature_size)
        self.rnn = nn.RNN(self.feature_size, hidden_size=n_hidden)
        self.layer_out = nn.Linear(n_hidden, self.vocab_size)

    def forward(self, x, hidden):
        x = x.transpose(0,1)
        x = self.encoder(x.long())
        x, hidden = self.rnn(x, hidden)
        x = x[-1]
        x = self.layer_out(x)
        return x

    def loss(self):
        loss = nn.CrossEntropyLoss()
        return loss

    def save_model(self, file_path):
        torch.save(self.state_dict(), file_path)

    def load_model(self, file_path):
        torch.load(self, file_path)