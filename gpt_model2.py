import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------
# Hyperparameters & Settings
# --------------------------
batch_size = 64  # how many independent sequences will we process in parallel?
block_size = 256  # maximum context length for predictions
max_iters = 5000  # total number of training iterations
eval_interval = 500  # evaluation frequency (in iterations)
learning_rate = 3e-4  # optimizer learning rate
eval_iters = 200  # number of batches to evaluate on
n_embd = 384  # embedding dimension
n_head = 6  # number of attention heads
n_layer = 6  # number of transformer blocks
dropout = 0.2  # dropout rate

# Set device (using 'mps' if available, else 'cpu')
device = 'mps' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(1337)

# --------------------------
# Data Loading & Processing
# --------------------------
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Create a mapping from characters to integers and vice-versa
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]  # encoder: string to list of ints
decode = lambda l: ''.join([itos[i] for i in l])  # decoder: list of ints to string

# Convert the text to a tensor of integers
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))  # first 90% for training, rest for validation
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    """
    Generate a batch of data for training or validation.

    Args:
        split (str): 'train' or 'val'
    Returns:
        tuple: (x, y) where x is the input tensor and y is the target tensor.
    """
    data_source = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_source) - block_size, (batch_size,))
    x = torch.stack([data_source[i:i + block_size] for i in ix])
    y = torch.stack([data_source[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model):
    """
    Estimate loss on both training and validation splits.
    """
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


# --------------------------
# Model Components
# --------------------------
class Head(nn.Module):
    """ One head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass for one self-attention head.
        """
        B, T, C = x.shape
        k = self.key(x)  # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        # Compute scaled dot-product attention
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)  # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v  # (B, T, head_size)
        return out


class MultiHeadAttention(nn.Module):
    """ Multiple self-attention heads in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """ A simple feedforward network """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Transformer block: self-attention followed by feedforward network """

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    """
    GPT-like language model for character-level prediction.
    """

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        Forward pass of the model.
        """
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            # Reshape logits and targets for computing cross-entropy loss
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = targets.view(-1)
            loss = F.cross_entropy(logits_flat, targets_flat)
        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0):
        """
        Generate text tokens from the model.

        Args:
            idx (torch.Tensor): (B, T) input tensor of token indices.
            max_new_tokens (int): Number of tokens to generate.
            temperature (float): Temperature for sampling; lower means less random.
        Returns:
            torch.Tensor: Tensor of shape (B, T + max_new_tokens) with generated tokens.
        """
        for i in range(max_new_tokens):
            # Only use the last block_size tokens as context
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            # Focus on the last time step and apply temperature scaling
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            if (i + 1) % 100 == 0:
                print(f"Generated {i + 1}/{max_new_tokens} tokens...")
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# --------------------------
# Training & Generation Routines
# --------------------------
def train(model, optimizer):
    """
    Train the model.
    """
    for iter in range(max_iters):
        # Evaluate loss on training and validation sets periodically
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model)
            print(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # Apply gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()


def main():
    """
    Main function to initialize the model, train, and generate text.
    """
    model = GPTLanguageModel().to(device)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"{total_params:.2f}M parameters")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Train the model
    train(model, optimizer)

    # Generate text after training
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = model.generate(context, max_new_tokens=500, temperature=0.8)
    print(decode(generated[0].tolist()))

    # Optionally, generate a longer output and save to file
    long_generated = model.generate(context, max_new_tokens=10000, temperature=0.8)
    with open('output3.txt', 'w') as f:
        f.write(decode(long_generated[0].tolist()))


if __name__ == '__main__':
    main()