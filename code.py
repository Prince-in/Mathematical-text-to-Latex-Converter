import torch
import torch.nn as nn
import torchvision.transforms as transforms
from collections import Counter
import pandas as pd
import random
from torch.utils.data import Dataset
from PIL import Image
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision.io import read_image
import os
# from nltk import word_tokenize
from collections import Counter


# cur_dir = "/kaggle/input/converting-handwritten-equations-to-latex-code/col_774_A4_2023"
cur_dir = os.getcwd()



class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.data = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.data.iloc[idx, 0])
        image = read_image(img_path)
        formula = self.data['formula'][idx]
        if self.transform:
            image = self.transform(image)
        return image, formula
    


class EncodeCNN(nn.Module):
    def __init__(self):
        super(EncodeCNN, self).__init__()
        self.flatten = nn.Flatten()
        self.stack = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AvgPool2d(kernel_size=3, stride=1)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.squeeze(0)
        x = self.stack(x)
        x = x.view(x.size(0), -1)
        return x


class DecodeLSTM(nn.Module):
    # takes context vector concatenated with previous word embedding as input and outputs next word
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(DecodeLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # context vector concatenated with previous word embedding
        # use LSTM Cell
        self.lstm = nn.LSTMCell(embedding_dim + hidden_dim, hidden_dim)
        # output layer -> transform hidden state to output (vocabulary space)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden, context_vector):
        embedded = self.embedding(x)
        # concatenate context vector with previous word embedding
        input = torch.cat((embedded, context_vector), dim=1)
        hidden = self.lstm(input, hidden)
        output = hidden[0]
        output = self.fc(output.squeeze(0))
        return output, hidden

    def init_hidden(self, batch_size, h0, c0):
        # initialize hidden state and cell state
        embedded = self.embedding(h0)
        # concatenate context vector with previous word embedding
        return (embedded, c0)


def load_data(batch_size=128):

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),#converts to [0,1] range
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))#converts to [-1,1] range
    ])
#     cur_dir = os.getcwd()

    training_data = CustomImageDataset(annotations_file=cur_dir+'/SyntheticData/train.csv',
                                        img_dir=cur_dir+'/SyntheticData/images/',
                                        transform=transform)

    test_data = CustomImageDataset(annotations_file=cur_dir+'/SyntheticData/test.csv',
                                            img_dir=cur_dir+'/SyntheticData/images/',
                                            transform=transform)


    # make embedding of the vocabulary
    vocab = Counter()
    vocab['<start>'] = len(vocab)
    vocab['<pad>'] = len(vocab)
    vocab['<end>'] = len(vocab)
    for formula in training_data.data['formula']:
        vocab.update(formula.split())
    idx_to_word = dict(enumerate(vocab))
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    vocab_size = len(vocab)
    # print(vocab_size)
    max_len = max(len(formula.split()) for formula in training_data.data['formula']) + 2 # +2 for <start> and <end>
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return train_dataloader, test_dataloader, vocab_size, idx_to_word, word_to_idx, max_len



def preprocess_formulas(formulas, word_to_idx):
    formulas = list(formulas)
    max_len = max(len(formula.split()) for formula in formulas) + 2
    m = len(formulas)
    for i in range(m):
        formula = formulas[i]
        formula_list = ['<start>'] + formula.split() + ['<end>']
        padding = ['<pad>'] * (max_len - len(formula_list))
        formula_list += padding
        formula_idx = [word_to_idx[word] for word in formula_list]
        formulas[i] = formula_idx
        formulas[i] = torch.tensor(formulas[i])
    return formulas, max_len






# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(device)
# Define hyperparameters

batch_size = 128
learning_rate = 0.0001
embedding_dim = 512
hidden_dim = 512
num_epochs = 10
teacher_forcing_ratio = 0.9

# Load data
train_dataloader, test_dataloader, vocab_size, idx_to_word, word_to_idx, max_len = load_data(batch_size)

# Initialize encoder and decoder

encoder = EncodeCNN().to(device)
decoder = DecodeLSTM(vocab_size, embedding_dim, hidden_dim).to(device)

# Initialize optimizer
# Use SGD optimizer

encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)


# Initialize loss function
criterion = nn.CrossEntropyLoss()

# Initialize lists to keep track of progress
losses = []
val_losses = []



# for i in range(10):
#     eval_BLEU()


def test():
    # set to eval mode
    encoder.eval()
    decoder.eval()

    # initialize lists to keep track of progress
    # take 1 example from train set and test the model's output
    images, formulas = next(iter(train_dataloader))
    images = images.to(device)
    formulas = torch.stack(formulas).to(device)
    encoder_out = encoder(images)
    decoder_input = torch.tensor([word_to_idx['<start>']] * batch_size).to(device)
    decoder_hidden = decoder.init_hidden(batch_size, decoder_input, encoder_out)
    decoder_hidden = (decoder_hidden[0].to(device), decoder_hidden[1].to(device))
    output_formula = []
    for t in range(1, max_len):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_out)
        top1 = decoder_output.argmax(1)
        output_formula.append(top1)
        decoder_input = top1

    # print output formula
    for i in range(len(output_formula)):
        flag = True
        for j in range(len(output_formula[i])):
            if (output_formula[i][j].item() != word_to_idx['<pad>']):
                flag = False
        if (flag):
            break
        print("Formula: \n")
        for j in range(len(formulas[i])):
            print(idx_to_word[formulas[i][j].item()], end=" ")
        print("\n\n")
        print("Output: \n")
        for j in range(len(output_formula[i])):
            print(idx_to_word[output_formula[i][j].item()], end=" ")
        print("\n\n")








def train(decoder, encoder, num_epochs=num_epochs):
    if os.path.exists(f'models/encoder-part_1.pth'):
        encoder.load_state_dict(torch.load(f'models/encoder-part_1.pth'))
    if os.path.exists(f'models/decoder-part_1.pth'):
        decoder.load_state_dict(torch.load(f'models/decoder-part_1.pth'))
    num_epochs = 100
    encoder.train()
    decoder.train()
    print("Training...")
    for epoch in range(num_epochs):
        total_loss = 0
        for i, (images, formulas) in enumerate(train_dataloader):
            size = len(images)
            images = images.to(device)
            formulas, max_len = preprocess_formulas(formulas, word_to_idx)
            formulas = torch.stack(formulas).to(device)
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            encoder_out = encoder(images)
            encoder_out = encoder_out.to(device)
            decoder_input = torch.tensor([word_to_idx['<start>']] * size).to(device)
            # put decoder hidden in device
            decoder_hidden = decoder.init_hidden(size, decoder_input, encoder_out)
            decoder_hidden = (decoder_hidden[0].to(device), decoder_hidden[1].to(device))
            loss = 0
            for t in range(1, max_len):
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_out)
                loss += criterion(decoder_output, formulas[:,t])
                teacher_force = random.random() < 0.9
                top1 = decoder_output.argmax(1)
                decoder_input = formulas[:, t] if teacher_force else top1

            loss = loss / max_len
            loss.backward()
            total_loss += loss.item()
            encoder_optimizer.step()
            decoder_optimizer.step()
            losses.append(loss.item())
            if i % 100 == 0:
                print('Epoch: {}/{}, Step: {}/{}, Loss: {}'.format(epoch + 1, num_epochs, i, len(train_dataloader), loss.item()))
            if i%100 == 0:
                torch.save(decoder.state_dict(), f"models/decoder-part_1.pth")
                torch.save(encoder.state_dict(), f"models/encoder-part_1.pth")
            
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_dataloader):.4f}")

        torch.save(decoder.state_dict(), f"models/decoder-part_1.pth")
        torch.save(encoder.state_dict(), f"models/encoder-part_1.pth")
        print("Training completed!")

train(decoder, encoder, num_epochs=num_epochs)



