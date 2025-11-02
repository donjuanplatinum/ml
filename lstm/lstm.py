from datasets import load_dataset

with open("/home/donjuan/git/datasets/rnn/chuci.txt", "r", encoding="utf-8") as f:
    text = f.read()
print(len(text), "characters")
print(text[:1000])


chars = sorted(list(set(text)))
vocab_size = len(chars)
print("Vocab size:", vocab_size)


stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    return ''.join([itos[i] for i in l])

import torch

data = torch.tensor(encode(text), dtype=torch.long)
print("Data shape:", data.shape)

split = int(0.9 * len(data))
train_data = data[:split]
val_data = data[split:]

block_size = 100  # 输入序列长度
def get_batch(split, batch_size=64):
    data_ = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_) - block_size, (batch_size,))
    x = torch.stack([data_[i:i+block_size] for i in ix])
    y = torch.stack([data_[i+1:i+block_size+1] for i in ix])
    return x, y
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, emb_size=256, hidden_size=1024, num_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_size)
        self.lstm = nn.LSTM(emb_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embed(x)
        out, hidden = self.lstm(x, hidden)
        logits = self.fc(out)
        return logits, hidden
device = "cuda" if torch.cuda.is_available() else "cpu"
model = LSTMModel(vocab_size).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
criterion = nn.CrossEntropyLoss()

for epoch in range(1000):
    x, y = get_batch('train')
    x, y = x.to(device), y.to(device)
    
    logits, _ = model(x)
    loss = criterion(logits.view(-1, vocab_size), y.view(-1))
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 2 == 0:
        print(f"Epoch {epoch}, loss: {loss.item():.4f}")

def generate(model, start_text, length=300):
    model.eval()
    input_eval = torch.tensor([encode(start_text)], dtype=torch.long).to(device)
    hidden = None
    result = list(start_text)

    for _ in range(length):
        logits, hidden = model(input_eval[:, -1:], hidden)
        probs = torch.softmax(logits[:, -1, :], dim=-1)
        next_id = torch.multinomial(probs, 1).item()
        result.append(itos[next_id])
        input_eval = torch.tensor([[next_id]], dtype=torch.long).to(device)

    return ''.join(result)

print(generate(model, "有一"))
