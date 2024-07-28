#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch.nn as nn
import nltk
import random
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
import numpy as np
from typing import Dict
import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from preprocessing_task_2 import prepare_data
import spacy
from collections import Counter


# In[2]:


nltk.download('punkt')
nltk.download('stopwords')
#!python3 -m spacy download en_core_web_sm
#!python3 -m spacy download fr_core_news_sm


# In[20]:


####### LOADING AND PREPROCESSING #############
english_data, french_data = prepare_data(fraction=0.01)


# In[3]:


# Functions to tokenize data and build vocab
spacy_fr = spacy.load('fr_core_news_sm')
spacy_en = spacy.load('en_core_web_sm')

def tokenize_fr(text):
    return [tok.text for tok in spacy_fr.tokenizer(text)]

def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

def build_vocab(sentences, vocab_size, word_tokenize):
    all_words = [word for sentence in sentences for word in word_tokenize(sentence)]
    word_counts = Counter(all_words)
    vocab = [word for word, _ in word_counts.most_common(vocab_size)]
    word2idx = {word: idx for idx, word in enumerate(vocab, start=4)}
    word2idx['<pad>'] = 0
    word2idx['<unk>'] = 1
    word2idx['<sos>'] = 2
    word2idx['<eos>'] = 3
    return word2idx


# In[ ]:





# In[4]:


# Collate function for padding
def collate_fn(batch):
    source_batch, target_batch = zip(*batch)
    source_batch_padded = pad_sequence(source_batch, padding_value=vocab_en['<pad>'], batch_first=True)
    target_batch_padded = pad_sequence(target_batch, padding_value=vocab_fr['<pad>'], batch_first=True)
    return source_batch_padded, target_batch_padded


# In[21]:


# Dataset preparation
class TranslationDataset(Dataset):
    def __init__(self, source_sentences, target_sentences, source_vocab, target_vocab, tokenizer_source, tokenizer_target):
        self.source_sentences = source_sentences
        self.target_sentences = target_sentences
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab
        self.tokenizer_source = tokenizer_source
        self.tokenizer_target = tokenizer_target
    
    def __len__(self):
        return len(self.source_sentences)
    
    def __getitem__(self, index):
        source_sentence = [self.source_vocab[token] if token in self.source_vocab else self.source_vocab['<unk>'] for token in self.tokenizer_source(self.source_sentences.iloc[index])]
        target_sentence = [self.target_vocab[token] if token in self.target_vocab else self.target_vocab['<unk>'] for token in self.tokenizer_target(self.target_sentences.iloc[index])]
        return torch.tensor(source_sentence, dtype=torch.long), torch.tensor(target_sentence, dtype=torch.long)




# In[22]:


# Example GloVe embedding file path and embedding dimension

def load_glove_embeddings(path: str, word2idx: Dict[str, int], embedding_dim: int) -> torch.Tensor:
    """
    Load GloVe embeddings from a specified file and align them with the given word index dictionary.

    Parameters:
    - path (str): The file path to the GloVe embeddings file.
    - word2idx (Dict[str, int]): A dictionary mapping words to their corresponding indices. This dictionary defines
      the position each word’s vector should occupy in the resulting embedding matrix.
    - embedding_dim (int): The dimensionality of the GloVe vectors (e.g., 50, 100, 200, 300).

    Returns:
    - torch.Tensor: A tensor of shape (len(word2idx), embedding_dim) containing the GloVe vectors aligned according to word2idx.
    """
    with open(path, 'r', encoding='utf-8') as f:
        # Initialize the embedding matrix with zeros
        #embeddings = np.zeros((len(word2idx), embedding_dim))
        #better approach: init with random 
        embeddings = np.random.uniform(-0.1, 0.1, (len(word2idx), embedding_dim))
        # Process each line in the GloVe file
        for line in f:
            values = line.split()
            word = values[0]
            
            # If the word is in the provided dictionary, update the corresponding row in embeddings
            if word in word2idx.keys():
                # Convert embedding values from strings to float32
                vector = np.asarray(values[1:], dtype='float32')
                # Place the vector in the correct index as per word2idx
                embeddings[word2idx[word]] = vector
            else:
                pass
    # Convert the numpy array to a PyTorch tensor
    return torch.from_numpy(embeddings)



def load_word2vec_embeddings(path, word2idx, embedding_dim):
    embeddings = np.random.uniform(-0.1, 0.1, (len(word2idx), embedding_dim))
    with open(path, 'r', encoding='latin1') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            if word in word2idx:
                try:
                    vector = np.asarray(values[1:], dtype='float32')
                    embeddings[word2idx[word]] = vector
                except ValueError:
                    print(f"Error converting values for word: {word}")
                    continue
    return torch.from_numpy(embeddings)






# In[23]:


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        trg_len = trg.shape[0]
        batch_size = trg.shape[1]
        trg_vocab_size = self.decoder.fc.out_features

        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        hidden = self.encoder(src)

        input = trg[0, :]

        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden)
            outputs[t] = output
            top1 = output.argmax(1)
            input = trg[t] if random.random() < teacher_forcing_ratio else top1

        return outputs




# In[24]:


class Encoder(nn.Module):
    def __init__(self, hidden_size, pretrained_embeddings, num_layers=2, bidirectional=True):
        super(Encoder, self).__init__()  # Ensure the class name here matches the class being defined
        if pretrained_embeddings.dtype != torch.float32:
            pretrained_embeddings = pretrained_embeddings.to(dtype=torch.float32)
        self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False)
        embed_size = pretrained_embeddings.shape[1]
        self.rnn = nn.GRU(embed_size, hidden_size, num_layers=num_layers, 
                          batch_first=False, bidirectional=bidirectional).float()

    def forward(self, input):
        embedded = self.embedding(input).float()
        _, hidden = self.rnn(embedded)
        if self.rnn.bidirectional:
            hidden = hidden.view(hidden.size(0)//2, 2, hidden.size(1), hidden.size(2))
            hidden = torch.sum(hidden, dim=1)
        return hidden

class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, output_size, pretrained_embeddings, num_layers=2):
        super(Decoder, self).__init__()  # Correct use of super()
        if pretrained_embeddings.dtype != torch.float32:
            pretrained_embeddings = pretrained_embeddings.to(dtype=torch.float32)
        self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False)
        self.rnn = nn.GRU(embed_size, hidden_size, num_layers=num_layers, batch_first=False).float()
        self.fc = nn.Linear(hidden_size, output_size).float()

    def forward(self, x, hidden):
        embedded = self.embedding(x).float()
        output, hidden = self.rnn(embedded, hidden)
        predicted = self.fc(output)
        return predicted, hidden


class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src = [src length, batch size]
        embedded = self.dropout(self.embedding(src))
        # embedded = [src length, batch size, embedding dim]
        outputs, hidden = self.rnn(embedded)  # no cell state in GRU!
        # outputs = [src length, batch size, hidden dim * n directions]
        # hidden = [n layers * n directions, batch size, hidden dim]
        # outputs are always from the top hidden layer
        return hidden

class Decoder(nn.Module):
    def __init__(self, output_dim, embedding_dim, hidden_dim, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, embedding_dim)
        self.rnn = nn.GRU(embedding_dim + hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(embedding_dim + hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, context):
        # input = [batch size]
        # hidden = [n layers * n directions, batch size, hidden dim]
        # context = [n layers * n directions, batch size, hidden dim]
        # n layers and n directions in the decoder will both always be 1, therefore:
        # hidden = [1, batch size, hidden dim]
        # context = [1, batch size, hidden dim]
        input = input.unsqueeze(0)
        # input = [1, batch size]
        embedded = self.dropout(self.embedding(input))
        # embedded = [1, batch size, embedding dim]
        emb_con = torch.cat((embedded, context), dim=2)
        # emb_con = [1, batch size, embedding dim + hidden dim]
        output, hidden = self.rnn(emb_con, hidden)
        # output = [seq len, batch size, hidden dim * n directions]
        # hidden = [n layers * n directions, batch size, hidden dim]
        # seq len, n layers and n directions will always be 1 in this decoder, therefore:
        # output = [1, batch size, hidden dim]
        # hidden = [1, batch size, hidden dim]
        output = torch.cat(
            (embedded.squeeze(0), hidden.squeeze(0), context.squeeze(0)), dim=1
        )
        # output = [batch size, embedding dim + hidden dim * 2]
        prediction = self.fc_out(output)
        # prediction = [batch size, output dim]
        return prediction, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        assert (
            encoder.hidden_dim == decoder.hidden_dim
        ), "Hidden dimensions of encoder and decoder must be equal!"

    def forward(self, src, trg, teacher_forcing_ratio):
        # src = [src length, batch size]
        # trg = [trg length, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        batch_size = trg.shape[1]
        trg_length = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        # tensor to store decoder outputs
        outputs = torch.zeros(trg_length, batch_size, trg_vocab_size).to(self.device)
        # last hidden state of the encoder is the context
        context = self.encoder(src)
        # context = [n layers * n directions, batch size, hidden dim]
        # context also used as the initial hidden state of the decoder
        hidden = context
        # hidden = [n layers * n directions, batch size, hidden dim]
        # first input to the decoder is the <sos> tokens
        input = trg[0, :]
        for t in range(1, trg_length):
            # insert input token embedding, previous hidden state and the context state
            # receive output tensor (predictions) and new hidden state
            output, hidden = self.decoder(input, hidden, context)
            # output = [batch size, output dim]
            # hidden = [1, batch size, hidden dim]
            # place predictions in a tensor holding predictions for each token
            outputs[t] = output
            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            # get the highest predicted token from our predictions
            top1 = output.argmax(1)
            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = trg[t] if teacher_force else top1
            # input = [batch size]
        return outputs

# In[25]:


def predict_1(model, sentence, source_vocab, target_vocab, tokenizer_source, tokenizer_target, max_length=50, device="cpu"):
    model.eval()  # Set the model to evaluation mode

    # Tokenize the input sentence
    tokens = tokenizer_source(sentence.lower())
    # Convert tokens to indices
    indices = [source_vocab.get(token, source_vocab['<unk>']) for token in tokens]
    input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)  # Add batch dimension
    # Pass through the encoder
    with torch.no_grad():
        encoder_hidden = model.encoder(input_tensor)
    # Initialize the decoder input and hidden state
    decoder_input = torch.tensor([[target_vocab['<sos>']]], dtype=torch.long).to(device)
    decoder_hidden = encoder_hidden#.unsqueeze(0)  # Add batch dimension back

    # Generate the output sequence
    output_tokens = []
    for _ in range(max_length):
        with torch.no_grad():
            decoder_output, decoder_hidden = model.decoder(decoder_input, decoder_hidden)

        topv, topi = decoder_output.topk(1)
        next_token = topi.squeeze().item()

        if next_token == target_vocab['<eos>']:
            break

        output_tokens.append(next_token)
        decoder_input = topi.squeeze(0)#.detach()#.unsqueeze(0)

    # Convert indices to tokens
    output_sentence = [list(target_vocab.keys())[list(target_vocab.values()).index(idx)] for idx in output_tokens]

    return ' '.join(output_sentence)


# In[43]:


import nltk
from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm 


def evaluate_bleu(model, data_loader, source_vocab, target_vocab, tokenizer_source, tokenizer_target, device, quick=False):
    """
    Evaluate the BLEU score for a translation model.

    Args:
        model: The translation model (Seq2Seq).
        data_loader: DataLoader for the dataset to evaluate.
        source_vocab: Vocabulary for the source language.
        target_vocab: Vocabulary for the target language.
        tokenizer_source: Tokenization function for the source language.
        tokenizer_target: Tokenization function for the target language.
        device: Device to run the model on ('cpu' or 'cuda').

    Returns:
        float: The BLEU score for the dataset.
    """
    model.eval()
    references = []
    hypotheses = []

    with torch.no_grad():
        counter = 0
        for source, target in tqdm(data_loader):
            if quick and counter == 5:
                break
            counter += 1
            # Process each sentence in the batch
            for i in range(source.size(0)):
                source_sentence = source[i]
                target_sentence = target[i]

                # Convert source tensor to sentence
                src_sent = ' '.join([list(source_vocab.keys())[list(source_vocab.values()).index(idx)] for idx in source_sentence if idx not in [source_vocab['<pad>'], source_vocab['<unk>'], source_vocab['<sos>'], source_vocab['<eos>']]])
                # Generate translation
                translation = predict_1(model, src_sent, source_vocab, target_vocab, tokenizer_source, tokenizer_target, device=device)
                
                # Convert target tensor to actual sentence
                ref_sent = [list(target_vocab.keys())[list(target_vocab.values()).index(idx)] for idx in target_sentence if idx not in [target_vocab['<pad>'], target_vocab['<unk>'], target_vocab['<sos>'], target_vocab['<eos>']]]
                #print(f"sentence: {src_sent}")
                #print(f"reference: {ref_sent}")
                #print(f"translation: {translation}")
                # Append to lists
                hypotheses.append(translation.split())
                references.append([ref_sent])

    # Calculate BLEU score
    return corpus_bleu(references, hypotheses)



# In[39]:


from rouge import Rouge

def evaluate_rouge(model, data_loader, source_vocab, target_vocab, tokenizer_source, tokenizer_target, device, quick=False):
    """
    Evaluate the ROUGE score for a translation model.

    Args:
        model: The translation model (Seq2Seq).
        data_loader: DataLoader for the dataset to evaluate.
        source_vocab: Vocabulary for the source language.
        target_vocab: Vocabulary for the target language.
        tokenizer_source: Tokenization function for the source language.
        tokenizer_target: Tokenization function for the target language.
        device: Device to run the model on ('cpu' or 'cuda').

    Returns:
        dict: A dictionary containing the ROUGE-1, ROUGE-2, and ROUGE-L scores.
    """
    model.eval()
    rouge = Rouge()
    references = []
    hypotheses = []

    with torch.no_grad():
        counter = 0
        for source, target in tqdm(data_loader):
            if quick and counter == 5:
                break
            counter += 1
            for i in range(source.size(0)):
                source_sentence = source[i]
                target_sentence = target[i]
                # More readable version
                src_sent = ' '.join(str(source_vocab.get(idx.item(), source_vocab['<unk>'])) for idx in source_sentence if idx not in [source_vocab['<pad>'], source_vocab['<unk>'], source_vocab['<sos>'], source_vocab['<eos>']])
                translation = predict_1(model, src_sent, source_vocab, target_vocab, tokenizer_source, tokenizer_target, device=device)
                ref_sent = ' '.join(str(target_vocab.get(idx.item(), target_vocab['<unk>'])) for idx in target_sentence if idx not in [target_vocab['<pad>'], target_vocab['<unk>'], target_vocab['<sos>'], target_vocab['<eos>']])                
                hypotheses.append(translation)
                references.append(ref_sent)

    return rouge.get_scores(hypotheses, references, avg=True)
                                     


# In[44]:


from tqdm import tqdm 

def train_4(model, train_loader,val_loader, optimizer, criterion, epochs=10, device="cpu"):
    model.train()  # Set the model to training mode
    for epoch in range(epochs):
        total_loss = 0
        for src, trg in tqdm(train_loader):
            src = src.to(device).long()
            trg = trg.to(device).long()

            optimizer.zero_grad()

            # Ensure the first token for the decoder is <sos> which is trg[:,0]
            # and the input to the decoder includes this <sos> token at the start
            # up to before the <eos> token at the end.
            decoder_input = trg[:, :-1]  # Excludes <eos> at the end
            target_output = trg[:, 1:]  # Excludes <sos> at the start

            # Pass the source and modified target to the model
            output = model(src, decoder_input)
            
            output = output.float()  # Ensure the output is in float format
            output_dim = output.shape[-1]

            # Flatten the output for computing the loss
            output = output.contiguous().view(-1, output_dim)
            target_output = target_output.contiguous().view(-1)

            # Compute the loss; we don't need to handle <eos> explicitly here as it's managed by the target sequence setup
            loss = criterion(output, target_output)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        average_loss = total_loss / len(train_loader)
        
        bleu_score = evaluate_bleu(model, val_loader, vocab_en, vocab_fr, tokenize_en, tokenize_fr, device, quick=True)
        print("bleu_score:",bleu_score)
        rouge_score = evaluate_rouge(model, val_loader, vocab_en, vocab_fr, tokenize_en, tokenize_fr, device, quick=True)
        print("rouge_1_score:",rouge_score["rouge-1"]["r"])
        print(predict_1(model, "Hello, this is a test", vocab_en, vocab_fr, tokenize_en, tokenize_fr, device=device))
        print(f'Epoch {epoch+1}/{epochs}, Loss: {average_loss:.4f}')

        


# In[45]:


import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torch.nn import CrossEntropyLoss, Module
from torch.optim import Adam
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from collections import Counter


# Define your device based on the availability
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Define the vocabulary size and embedding dimension
vocab_size = 10000

# Build vocabularies based on the training set only
X_train, X_test, y_train, y_test = train_test_split(english_data, french_data, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25)  # 0.25 x 0.8 = 0.2

# Building vocabularies using only the training data to prevent information leakage
vocab_en = build_vocab(X_train["text"], vocab_size, tokenize_en)
vocab_fr = build_vocab(y_train["text"], vocab_size, tokenize_fr)
print(f"size of english vocab {len(vocab_en)}, size of french vocab {len(vocab_fr)}")

#Load embeddings glove
#embedding_dim = 300
#vocab_embeddings_en = load_glove_embeddings('glove.6B/glove.6B.300d.txt', vocab_en, embedding_dim)
#vocab_embeddings_fr = load_glove_embeddings('fasttext/cc.fr.300.vec', vocab_fr, embedding_dim)

#Load embeddings word2vec
embedding_dim = 100
vocab_embeddings_en = load_word2vec_embeddings('word2vec/english.txt', vocab_en, embedding_dim)
vocab_embeddings_fr = load_word2vec_embeddings('word2vec/france.txt', vocab_fr, embedding_dim)
print(f"size of english embeddings {vocab_embeddings_en.shape}, size of french embeddings {vocab_embeddings_fr.shape}")


input_dim = len(vocab_en)
output_dim = len(vocab_fr)
hidden_dim = 512
encoder_dropout = 0.5
decoder_dropout = 0.5

encoder = Encoder(
    input_dim,
    embedding_dim,
    hidden_dim,
    encoder_dropout,
)

decoder = Decoder(
    output_dim,
    embedding_dim,
    hidden_dim,
    decoder_dropout,
)

model = Seq2Seq(encoder, decoder, device).to(device)

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.normal_(param.data, mean=0, std=0.01)


model.apply(init_weights)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f"The model has {count_parameters(model):,} trainable parameters")

# In[46]:


# Instantiate the model components
#hidden_size = 1024
#encoder = Encoder(hidden_size=hidden_size, pretrained_embeddings=vocab_embeddings_en, num_layers=1, bidirectional=False)
#decoder = Decoder(embed_size=embedding_dim, hidden_size=hidden_size, output_size=len(vocab_fr), pretrained_embeddings=vocab_embeddings_fr, num_layers=1)

#model = Seq2Seq(encoder, decoder, device=device)
model = model.to(device)

# Define the optimizer and loss function
optimizer = Adam(model.parameters(), lr=0.001)
criterion = CrossEntropyLoss(ignore_index=vocab_fr['<pad>']).to(device)


def translate_sentence(
    sentence,
    model,
    source_tokenizer,
    en_vocab,
    fr_vocab,
    lower,
    sos_token,
    eos_token,
    device,
    max_output_length=25,
):
    model.eval()
    with torch.no_grad():
        if isinstance(sentence, str):
            tokens = [token for token in source_tokenizer(sentence)]
        else:
            tokens = [token for token in sentence]
        if lower:
            tokens = [token.lower() for token in tokens]
        tokens = [sos_token] + tokens + [eos_token]
        ids = [en_vocab[token] for token in tokens]
        tensor = torch.LongTensor(ids).unsqueeze(-1).to(device)
        context = model.encoder(tensor)
        hidden = context
        inputs = [fr_vocab[sos_token]]
        for _ in range(max_output_length):
            inputs_tensor = torch.LongTensor([inputs[-1]]).to(device)
            output, hidden = model.decoder(inputs_tensor, hidden, context)
            predicted_token = output.argmax(-1).item()
            inputs.append(predicted_token)
            if predicted_token == fr_vocab[eos_token]:
                break
        tokens = [fr_vocab[token] for token in inputs]
    return tokens

def train_fn(
    model, data_loader, optimizer, criterion, clip, teacher_forcing_ratio, device
):
    model.train()
    epoch_loss = 0
    for src, trg in tqdm(data_loader):
        src = src.to(device).long().T
        trg = trg.to(device).long().T
        # src = [src length, batch size]
        # trg = [trg length, batch size]
        optimizer.zero_grad()
        output = model(src, trg, teacher_forcing_ratio)
        # output = [trg length, batch size, trg vocab size]
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        # output = [(trg length - 1) * batch size, trg vocab size]
        #trg = trg[1:].view(-1)
        trg = trg[1:].reshape(-1)
        # trg = [(trg length - 1) * batch size]
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(data_loader)

# Create datasets for training and validation
train_dataset = TranslationDataset(X_train['text'], y_train['text'], vocab_en, vocab_fr, tokenize_en, tokenize_fr)
val_dataset = TranslationDataset(X_val['text'], y_val['text'], vocab_en, vocab_fr, tokenize_en, tokenize_fr)

# Create DataLoaders for training and validation
train_loader = DataLoader(train_dataset, batch_size=32, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=32, collate_fn=collate_fn)


n_epochs = 10
clip = 1.0
teacher_forcing_ratio = 0.5

best_valid_loss = float("inf")

def evaluate_fn(model, data_loader, criterion, device):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for src, trg in tqdm(data_loader):
            src = src.to(device).long().T
            trg = trg.to(device).long().T
            # src = [src length, batch size]
            # trg = [trg length, batch size]
            output = model(src, trg, 0)  # turn off teacher forcing
            # output = [trg length, batch size, trg vocab size]
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            # output = [(trg length - 1) * batch size, trg vocab size]
            #trg = trg[1:].view(-1)
            trg = trg[1:].reshape(-1)
            # trg = [(trg length - 1) * batch size]
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(data_loader)

for epoch in range(n_epochs):
    train_loss = train_fn(
        model,
        train_loader,
        optimizer,
        criterion,
        clip,
        teacher_forcing_ratio,
        device,
    )
    valid_loss = evaluate_fn(
        model,
        val_loader,
        criterion,
        device,
    )
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), "model.pt")

    lower = True
    sos_token = "<sos>"
    eos_token = "<eos>"
    translation = translate_sentence(
        "i am a student",
        model,
        tokenize_en,
        vocab_en,
        vocab_fr,
        lower,
        sos_token,
        eos_token,
        device,
    )
    print(f"\tTrain Loss: {train_loss:7.3f} | Train PPL: {np.exp(train_loss):7.3f}")
    print(f"\tValid Loss: {valid_loss:7.3f} | Valid PPL: {np.exp(valid_loss):7.3f}")


# In[ ]:


print(predict_1(model, "I am a student", vocab_en, vocab_fr, tokenize_en, tokenize_fr, device=device))
print(predict_2(model, "I am a book", vocab_en, vocab_fr, tokenize_en, tokenize_fr, device=device))


# In[ ]:


#############################################################################################################


# In[140]:


# Example GloVe embedding file path and embedding dimension

def check_missing_words_in_embeddings(path: str, word2idx: Dict[str, int], embedding_dim: int) -> torch.Tensor:
    print(len(word2idx))
    missing_word_counter = 0
    
    
        
    glove_words = set()
    with open(path, 'r', encoding='utf-8') as f:
        # Process each line in the GloVe file to collect all GloVe words
        for line in f:
            values = line.split()
            word = values[0]
            glove_words.add(word)
    
    for word in word2idx:
        if word not in glove_words:
            missing_word_counter+=1
    

    print(missing_word_counter)
    # Convert the numpy array to a PyTorch tensor

check_missing_words_in_embeddings('glove.6B/glove.6B.100d.txt', vocab_fr, embedding_dim)

check_missing_words_in_embeddings('fasttext/cc.fr.300.vec', vocab_fr, 300) 


# In[141]:


"""
Relevant notes:
If a word in  the vocabulary (word2idx) doesn’t exist in the GloVe or FastText embeddings file, 
its embedding vector would not be updated and would remain as initially set by a zero vector. 

update:
not found in vocubulary words are initialized with random vector


"""


# In[207]:


train_dataset[5]


# In[208]:


def get_key_by_value(dictionary, search_value):
    for key, value in dictionary.items():
        if value == search_value:
            return key
    return None  # If the value is not found, return None or raise an exception


list_en= [ 687,  124,    7,  149,    4,  377, 1420,  195,    5,    7,  149,   13,
          926,  688,   66,  689,    9,   13,   50,   11,  452,   30,    4,  322,
            8]

print("english:")

res= ""
for i in list_en:
    res= res + " "+  get_key_by_value(vocab_en, i)
print(res)
    

##########


list_fr= [1479,   53,  265,   14,    7,  943,   46,  944,    4,   35,   37,   76,
          197,    9,  714,    4,   35,   21,  579,  103, 1480,   93,   11,  266,
            6]

print("french:")

res= ""
for i in list_fr:
    res= res + " "+  get_key_by_value(vocab_fr, i)
print(res)


# In[199]:


english_data


# In[156]:


french_data


# In[ ]:





# In[177]:


X_train, X_test, y_train, y_test = train_test_split(english_data, french_data, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25)  # 0.25 x 0.8 = 0.2


# In[181]:


X_train["text"][5]


# In[182]:


y_train["text"][5]


# In[ ]:





# In[ ]:




