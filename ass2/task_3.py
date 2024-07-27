#!/usr/bin/env python
# coding: utf-8

# In[3]:


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



####### LOADING AND PREPROCESSING #############
english_data, french_data = prepare_data()


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


# Collate function for padding
def collate_fn(batch):
    source_batch, target_batch = zip(*batch)
    source_batch_padded = pad_sequence(source_batch, padding_value=vocab_en['<pad>'], batch_first=True)
    target_batch_padded = pad_sequence(target_batch, padding_value=vocab_fr['<pad>'], batch_first=True)
    return source_batch_padded, target_batch_padded





def load_embeddings_and_create_index(path):
    word_to_idx = {}
    idx = 0
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            word_to_idx[word] = idx
            idx += 1
    return word_to_idx



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
        embeddings = np.zeros((len(word2idx), embedding_dim))
        
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
    
    # Convert the numpy array to a PyTorch tensor
    return torch.from_numpy(embeddings)




class Encoder(nn.Module):
    def __init__(self, hidden_size, pretrained_embeddings):
        
        """
        Initialize the Encoder with pre-trained embeddings and a GRU layer.

        Parameters:
            hidden_size (int): The number of features in the hidden state of the GRU.
            pretrained_embeddings (torch.Tensor): A tensor containing the pre-trained word embeddings.
        """
        super(Encoder, self).__init__()
        # Ensure that the pretrained embeddings are of type float32
        if pretrained_embeddings.dtype != torch.float32:
            pretrained_embeddings = pretrained_embeddings.to(dtype=torch.float32)
        self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False)
        embed_size = pretrained_embeddings.shape[1]  # Embedding size is the second dimension of the embeddings tensor
        self.rnn = nn.GRU(embed_size, hidden_size, batch_first=True).float()  # Ensure GRU is initialized as float32

    def forward(self, input):
        #print(input.size())
        """
        Forward pass of the encoder which processes the input sequence.

        Parameters:
            input (torch.Tensor): The input sequence tensor, which should be indexed by batch.

        Returns:
            hidden (torch.Tensor): The hidden state of the GRU, representing the encoded information of the input.
        """
        embedded = self.embedding(input).float()  # Ensure embedding outputs float32
        _, hidden = self.rnn(embedded)
        return hidden

class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, output_size, pretrained_embeddings):
        """
        Initialize the Decoder with pre-trained embeddings, a GRU layer, and a linear output layer.

        Parameters:
            embed_size (int): The size of each embedding vector.
            hidden_size (int): The number of features in the hidden state of the GRU.
            output_size (int): The size of the output vocabulary.
            pretrained_embeddings (torch.Tensor): A tensor containing the pre-trained word embeddings.

        """
        
        super(Decoder, self).__init__()
        # Ensure that the pretrained embeddings are of type float32
        if pretrained_embeddings.dtype != torch.float32:
            pretrained_embeddings = pretrained_embeddings.to(dtype=torch.float32)
        self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False)
        self.rnn = nn.GRU(embed_size, hidden_size, batch_first=True).float()  # Ensure GRU is initialized as float32
        self.fc = nn.Linear(hidden_size, output_size).float()  # Ensure Linear is initialized as float32

    def forward(self, x, hidden):
        """
        Forward pass of the decoder that processes one timestep of the sequence.

        Parameters:
            x (torch.Tensor): The input tensor for the current timestep.
            hidden (torch.Tensor): The hidden state from the last timestep.

        Returns:
            predicted (torch.Tensor): The output logits for the next word in the sequence.
            hidden (torch.Tensor): The updated hidden state.
        """
        embedded = self.embedding(x).float()  # Ensure embedding outputs float32
        output, hidden = self.rnn(embedded, hidden)
        predicted = self.fc(output)
        return predicted, hidden
    
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, source, target):

        """
        Forward pass of the Seq2Seq model which processes the entire input and target sequence.

        Parameters:
            source (torch.Tensor): The input sequence tensor.
            target (torch.Tensor): The target sequence tensor used during training.

        Returns:
            outputs (torch.Tensor): The output from the decoder for each step in the sequence.
        """
        hidden = self.encoder(source)
        outputs, _ = self.decoder(target, hidden)
        return outputs




def train(model, loader, optimizer, criterion, epochs=10, device="cpu"):
    model.train()  # Set the model to training mode
    for epoch in range(epochs):
        total_loss = 0
        for src, trg in loader:
            # Move tensors to the correct device and ensure they are long type for indexing operations
            src = src.to(device).long()  # Correct type for embedding layer
            trg = trg.to(device).long()  # Correct type for embedding layer
            

            optimizer.zero_grad()

            # Forward pass: The decoder's input is all except the last word
            output = model(src, trg[:, :-1])  
            
            # Since output will be in float (from linear layers, and GRU output), ensure it's float32 if not already
            output = output.float()

            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)  # Target doesn't include the first <sos> token

            loss = criterion(output, trg)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        average_loss = total_loss / len(loader)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {average_loss:.4f}')
        
        

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

vocab_size = 10000
# TODO: Do we build vocab on the entire dataset or just the training set?
#word2idx_pre_embeddings = load_embeddings_and_create_index('glove.6B/glove.6B.100d.txt') TODO remove
vocab_en = build_vocab(english_data["text"], vocab_size, tokenize_en)
vocab_fr = build_vocab(french_data["text"], vocab_size, tokenize_fr)

X_train, X_test, y_train, y_test = train_test_split(english_data, french_data, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

embedding_dim = 100

# Load embeddings
vocab_embeddings_en = load_glove_embeddings('glove.6B/glove.6B.100d.txt', vocab_en, embedding_dim)
vocab_embeddings_fr = load_glove_embeddings('glove.6B/glove.6B.100d.txt', vocab_fr, embedding_dim)

# Model instantiation
hidden_size = 1024
encoder = Encoder(hidden_size=hidden_size, pretrained_embeddings=vocab_embeddings_en)
decoder = Decoder(embed_size=embedding_dim, hidden_size=hidden_size, output_size=len(vocab_fr), pretrained_embeddings=vocab_embeddings_fr)
model = Seq2Seq(encoder, decoder)
model = model.to(device)


optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = CrossEntropyLoss(ignore_index=vocab_fr['<pad>']).to(device)  # Move the loss function to the device

dataset = TranslationDataset(english_data['text'], french_data['text'], vocab_en, vocab_fr, tokenizer_source=tokenize_en, tokenizer_target=tokenize_fr)
loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)


train(model, loader, optimizer, criterion, epochs=10, device=device)


def predict(model, sentence, source_vocab, target_vocab, tokenizer_source, tokenizer_target, max_length=50, device="cpu"):
    model.eval()  # Set the model to evaluation mode

    # Tokenize the input sentence
    tokens = tokenizer_source(sentence.lower())

    # Convert tokens to indices
    indices = [source_vocab.get(token, source_vocab['<unk>']) for token in tokens]
    print(f"indices: {indices}")
    print(f"words: {[list(source_vocab.keys())[list(source_vocab.values()).index(idx)] for idx in indices]}")
    # Prepare the input tensor
    input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)  # Add batch dimension
    print(f"input tensor size: {input_tensor.size()}")
    # Pass through the encoder
    with torch.no_grad():
        encoder_hidden = model.encoder(input_tensor)

    print(f"encoder hidden size: {encoder_hidden.size()}")

    # Initialize the decoder input and hidden state
    decoder_input = torch.tensor([[target_vocab['<sos>']]], dtype=torch.long).to(device)
    print(f"decoder input size: {decoder_input.size()}")
    decoder_hidden = encoder_hidden#.unsqueeze(0)  # Add batch dimension back
    print(f"decoder hidden size: {decoder_hidden.size()}")


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


predict(model, "I am a book", vocab_en, vocab_fr, tokenize_en, tokenize_fr, device=device)

# do some predictions
def predict(model, sentence, source_vocab, target_vocab, tokenizer_source, tokenizer_target, device="cpu"):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        source = torch.tensor([source_vocab[token] if token in source_vocab else source_vocab['<unk>'] for token in tokenizer_source(sentence)], dtype=torch.long).unsqueeze(0).to(device)
        target = torch.tensor([target_vocab['<sos>']], dtype=torch.long).unsqueeze(0).to(device)
        hidden = model.encoder(source)
        outputs = []
        for _ in range(20):  # Limit the length of the generated sequence
            output, hidden = model.decoder(target, hidden)
            output = output.squeeze(0)
            topv, target = output.topk(1)
            #target = topi.squeeze(0)#.detach()
            if target.item() == target_vocab['<eos>']:
                break
            outputs.append(target.item())
        translated = ' '.join([list(target_vocab.keys())[list(target_vocab.values()).index(idx)] for idx in outputs])
        return translated

predict(model, "I am a book", vocab_en, vocab_fr, tokenize_en, tokenize_fr, device=device)



