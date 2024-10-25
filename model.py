import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import Dataset, DataLoader

class LMulMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, mantissa_bits=3):
        super(LMulMultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.mantissa_bits = mantissa_bits

        self.d_k = d_model // num_heads
        self.w_q = LMulLinear(d_model, d_model, mantissa_bits)
        self.w_k = LMulLinear(d_model, d_model, mantissa_bits)
        self.w_v = LMulLinear(d_model, d_model, mantissa_bits)
        self.w_o = LMulLinear(d_model, d_model, mantissa_bits)

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn_probs = F.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, v)
        return output

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        # Project inputs to query, key, and value tensors
        q = self.w_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Apply attention
        attn_output = self.scaled_dot_product_attention(q, k, v, mask)
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        # Final linear layer
        return self.w_o(attn_output)

class LMulTransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout, mantissa_bits=3):
        super(LMulTransformerBlock, self).__init__()
        self.attn = LMulMultiHeadAttention(d_model, num_heads, mantissa_bits)
        self.ff = nn.Sequential(
            LMulLinear(d_model, d_ff, mantissa_bits),
            nn.ReLU(),
            LMulLinear(d_ff, d_model, mantissa_bits)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output = self.attn(x, x, x, mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        ff_output = self.ff(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)
        return x

class LMulLanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout, mantissa_bits=3):
        super(LMulLanguageModel, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = nn.Parameter(self._init_positional_encoding(max_seq_length, d_model), requires_grad=False)
        self.transformer_blocks = nn.ModuleList([
            LMulTransformerBlock(d_model, num_heads, d_ff, dropout, mantissa_bits) for _ in range(num_layers)
        ])
        self.fc_out = LMulLinear(d_model, vocab_size, mantissa_bits)
        self.dropout = nn.Dropout(dropout)

    def _init_positional_encoding(self, max_seq_length, d_model):
        position = torch.arange(0, max_seq_length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_seq_length, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x, mask=None):
        x = self.token_embedding(x) + self.positional_encoding[:, :x.size(1), :]
        x = self.dropout(x)

        for block in self.transformer_blocks:
            x = block(x, mask)

        logits = self.fc_out(x)
        return logits

class LMulLinear(nn.Module):
    def __init__(self, in_features, out_features, mantissa_bits=3):
        super(LMulLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mantissa_bits = mantissa_bits
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def l_mul(self, x, y):
        # Convert x and y to mantissa and exponent
        mant_x, exp_x = torch.frexp(x)
        mant_y, exp_y = torch.frexp(y)

        shift_amount = self.mantissa_bits
        int_mant_x = torch.round(mant_x * (2 ** shift_amount)).to(torch.int32)
        int_mant_y = torch.round(mant_y * (2 ** shift_amount)).to(torch.int32)

        # Expand dimensions
        int_mant_x = int_mant_x.unsqueeze(-1)  # Shape: [batch_size, seq_length, in_features, 1]
        int_mant_y = int_mant_y.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, in_features, out_features]
        exp_x = exp_x.unsqueeze(-1)  # Shape: [batch_size, seq_length, in_features, 1]
        exp_y = exp_y.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, in_features, out_features]

        # Compute products
        prod_int_mant = int_mant_x * int_mant_y  # Shape: [batch_size, seq_length, in_features, out_features]
        prod_exp = exp_x + exp_y

        # Adjust for shifted mantissas
        total_shift = 2 * shift_amount

        # Convert products back to floats
        prod_float = torch.ldexp(prod_int_mant.float(), prod_exp - total_shift)

        # Align exponents before summing
        max_exp, _ = torch.max(prod_exp, dim=2, keepdim=True)
        aligned_mant = prod_float * torch.ldexp(torch.ones_like(prod_float), prod_exp - max_exp)
        sum_mant = torch.sum(aligned_mant, dim=2)
        result_exp = max_exp.squeeze(2)

        # Final result
        result = torch.ldexp(sum_mant, result_exp)

        return result

    def forward(self, input):
        output = self.l_mul(input, self.weight.t())
        if self.bias is not None:
            output = output + self.bias
        return output

# Example dataset and model setup
training_text = "Ceci est un exemple de texte pour l'entraînement de notre modèle de langage."

class SimpleTokenizer:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
    
    def build_vocab(self, text):
        words = text.split()
        unique_words = set(words)
        for idx, word in enumerate(unique_words):
            self.word2idx[word] = idx
            self.idx2word[idx] = word
    
    def encode(self, text):
        words = text.split()
        return [self.word2idx[word] for word in words]
    
    def decode(self, tokens):
        return ' '.join([self.idx2word[token] for token in tokens if token in self.idx2word])

tokenizer = SimpleTokenizer()
tokenizer.build_vocab(training_text)

vocab_size = len(tokenizer.word2idx)
d_model = 16  # smaller model for testing
num_heads = 2
num_layers = 1
d_ff = 64
max_seq_length = 5
dropout = 0.1
mantissa_bits = 3

class TextDataset(Dataset):
    def __init__(self, text, seq_length, tokenizer):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.tokens = self.tokenizer.encode(text)
        # Duplicate tokens if necessary
        self.tokens = self.tokens + self.tokens

    def __len__(self):
        return len(self.tokens) - self.seq_length

    def __getitem__(self, idx):
        x = self.tokens[idx: idx + self.seq_length]
        y = self.tokens[idx + 1: idx + self.seq_length + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

dataset = TextDataset(training_text, max_seq_length, tokenizer)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)  # smaller batch size

model = LMulLanguageModel(
    vocab_size, d_model, num_heads, num_layers, d_ff,
    max_seq_length, dropout, mantissa_bits
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

num_epochs = 3  # fewer epochs for testing
model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for x_batch, y_batch in dataloader:
        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs.view(-1, vocab_size), y_batch.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

def generate_text(model, tokenizer, prompt, max_length=10):
    model.eval()
    tokens = tokenizer.encode(prompt)
    generated = tokens.copy()

    for _ in range(max_length):
        input_ids = torch.tensor([generated[-max_seq_length:]], dtype=torch.long)
        outputs = model(input_ids)
        predictions = outputs[0, -1, :]
        predicted_id = torch.argmax(predictions).item()
        generated.append(predicted_id)
        if len(generated) > max_seq_length:
            generated = generated[-max_seq_length:]
    
    return tokenizer.decode(generated)

prompt = "Ceci est un"
generated_text = generate_text(model, tokenizer, prompt)
print("Generated text:")
print(generated_text)
