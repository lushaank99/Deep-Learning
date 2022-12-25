import torch
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch import nn, optim, cuda
from torch.utils import data as torch_data
from torchvision import datasets, transforms
from keras import layers, models, initializers

plt.rcParams['image.cmap'] = 'gray'
sns.set()
device = 'cuda' if cuda.is_available() else 'cpu'
warnings.filterwarnings('ignore')

def train_epoch(epoch_no, model, optimizer, criterion, data_l):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    total = 0
    
    for i, (features, labels) in enumerate(data_l):
        total += labels.size(0)
        
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        output = model(features)
        loss = criterion(output, labels)
        _, predicted = torch.max(output, dim=-1)

        epoch_acc += (predicted == labels).sum().item()
        epoch_loss += loss.item()

        loss.backward()
        optimizer.step()
    
    return epoch_loss / total, epoch_acc / total

def train(model, optimizer, criterion, train_data, epochs=10):
    losses = []
    accs = []
    for epoch in range(epochs):
        l, a = train_epoch(epoch, model, optimizer, criterion, train_data)
        losses.append(l)
        accs.append(a)
        
        print(f"Epoch: {epoch}")
        print(f"\tLoss: {l}")
        print(f"\tAccuracy: {a}")
    
    return losses, accs

words = []
embed_vec_dicts = dict()
embed_matrix = []
with open("glove.6B.100d.txt", mode='r', encoding='utf-8') as f:
    for line in f:
        split_line = line.split()
        word = split_line.pop(0)
        word_vec = [eval(val) for val in split_line]
        words.append(word.lower())
        embed_vec_dicts[word] = word_vec
        embed_matrix.append(word_vec)

embed_matrix = np.matrix(embed_matrix)
print(embed_matrix.shape)
i_to_word = dict(enumerate(words))
word_to_i = {val: key for key, val in i_to_word.items()}

def get_index(x):
    out = []
    for w in x:
        if w in word_to_i.keys():
            out.append(word_to_i[w])
    
    return out

imdb = pd.read_csv("imdb_master.csv", header=0, encoding='cp1252', usecols=["type", "review", "label"])
imdb = imdb.loc[imdb.label != 'unsup', :]# removing the unsup class
imdb.review = imdb.review.str.lower()
imdb['inputs'] = imdb.review.str.split().apply(get_index)
imdb['is_pos_class'] = imdb.label.apply(lambda x: 1 if x == 'pos' else 0)
imdb = imdb.join(pd.get_dummies(imdb.label))
imdb.neg = imdb.neg.apply(lambda x: 1 - x)
imdb.pos = imdb.pos.apply(lambda x: 1 - x)
imdb.head()

imdb_train = imdb.loc[imdb.type == 'train', :]
imdb_test = imdb.loc[imdb.type == 'test', :]
imdb_train.index = range(imdb_train.shape[0])
imdb_test.index = range(imdb_test.shape[0])
print(f"Train: {imdb_train.shape}\nTest: {imdb_test.shape}")

fig, ax = plt.subplots(1, 2, figsize=(6.4 * 2, 4.8))
_ = sns.countplot(x='label', data=imdb_train, ax=ax[0])
_ = sns.countplot(x='label', data=imdb_test, ax=ax[1])

class IMDB(torch_data.Dataset):
    def __init__(self, data):
        self.data = data.reset_index().drop(['index'], axis=1)
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
#         print(idx)
#         print(f"Stuff: {torch.LongTensor(self.data.loc[idx, 'inputs'])}")
#         print(f"Stuff Again: {(self.data.loc[idx, 'is_pos_class'])}") # self.data.loc[idx, slice('neg', 'pos')]
        return torch.LongTensor(self.data.loc[idx, 'inputs']), self.data.loc[idx, 'is_pos_class']


train_set = IMDB(imdb_train)
test_set = IMDB(imdb_test)

train_loader = torch_data.DataLoader(train_set, shuffle=True, batch_size=1)
test_loader = torch_data.DataLoader(test_set, shuffle=True, batch_size=1)

class SimpleLSTM(nn.Module):
    def __init__(self, num_classes, vocab_size, embedding_dim, hidden_dim, embed_mtx, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.hid = hidden_dim
        
        self.embedding = nn.Embedding.from_pretrained(torch.Tensor(embed_mtx))
        self.embedding.weight.requires_grad = False
        self.embedding.to(device)
        
        self.lstm1 = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim).to(device)
        self.flatten = nn.Flatten().to(device)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2).to(device)
        self.output_layer = nn.Linear(hidden_dim // 2, num_classes).to(device)
        self.relu = nn.ReLU().to(device)
        self.tanh = nn.Tanh().to(device)
        self.softmax = nn.Softmax().to(device)
    
    def forward(self, x):
#         if type(x) != torch.Tensor:
#             x = torch.FloatTensor(x).to(device)
        x = self.embedding(x)
        out, (hidden, context) = self.lstm1(x)
        x = self.tanh(out)
#         print(f"{x.shape}")
        x = self.flatten(x)
#         print(f"X: {x.size(1)}")
        x = nn.Linear(x.size(1), self.hid // 2).to(device)(x)
        x = self.relu(x)
        x = self.softmax(self.output_layer(x))
        return x


simple_lstm = SimpleLSTM(len(pd.unique(imdb_train.label)), *embed_matrix.shape, 20, embed_matrix)
adam_opt = optim.Adam(simple_lstm.parameters())
cel = nn.CrossEntropyLoss()

ls_, as_ = train(simple_lstm, adam_opt, cel, train_loader, epochs=2)

fig, ax = plt.subplots(1, 2, figsize=(6.4 * 2, 4.8))
_ = sns.lineplot(range(len(ls_)), ls_, ci=None, ax=ax[0])
_ = ax[0].set_title("Loss")
_ = sns.lineplot(range(len(as_)), as_, ci=None, ax=ax[1])
_ = ax[1].set_title("Accuracy")