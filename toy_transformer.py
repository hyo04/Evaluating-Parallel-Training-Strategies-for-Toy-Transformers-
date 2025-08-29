import torch
import torch.nn as nn
import torch.nn.functional as F


# --- Config ---
class TransformerConfig:
    def __init__(self,
                 vocab_size=1000,       #1000
                 seq_len=256,           #256
                 d_model=512,           #512
                 n_heads=4,             #8
                 n_layers=35,           #12
                 ffn_mult=4,            #4
                 dropout=0.1,           #0.1
                 batch_size=32):        #32
      
      self.vocab_size = vocab_size
      self.seq_len = seq_len
      self.d_model = d_model
      self.n_heads = n_heads
      self.n_layers = n_layers
      self.ffn_mult = ffn_mult
      self.dropout = dropout
      self.batch_size = batch_size

# --- Embeddings ---
class TokenPositionalEmbedding(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        #embedding layer for the tokenID
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        #embedding layer for the position 
        self.pos_embedding = nn.Embedding(config.seq_len, config.d_model)

    def forward(self, token_ids):
        batch_size, seq_len = token_ids.shape
        #device=token_ids.device makes sure that the position matrix is created on the same device as the token_ids 
        #unsqueeze(0) adds a new dimension to match the token embedding dimension 
        positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        token_embeds = self.token_embedding(token_ids)
        pos_embeds = self.pos_embedding(positions)
        return token_embeds + pos_embeds
        

# --- Multi-Head Attention ---
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
         #split the vector into differerent heads -> size per head is d_model/heads 
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        self.head_dim = self.d_model // self.n_heads

        #linear layers to produce q,k,v
        self.q_linear = nn.Linear(self.d_model, self.d_model)
        self.k_linear = nn.Linear(self.d_model, self.d_model)
        self.v_linear = nn.Linear(self.d_model, self.d_model)

        #linear layer for the output after concatenating heads 
        self.out_linear = nn.Linear(self.d_model, self.d_model)


    def forward(self, x, mask=None):
        
        #compute Q,K,V weights 
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)

        batch_size, seq_len, d_model = x.shape
        head_dim = self.head_dim
        n_heads = self.n_heads

        #split into multiple heads -> reshape 
        x = x.view(batch_size, seq_len, n_heads, head_dim)
        x = x.permute(0,2,1,3)
        q = q.view(batch_size, seq_len, n_heads, head_dim)
        q = q.permute(0,2,1,3)
        k = k.view(batch_size, seq_len, n_heads, head_dim)
        k = k.permute(0,2,1,3)
        v = v.view(batch_size, seq_len, n_heads, head_dim)
        v = v.permute(0,2,1,3)

        #each scores[b,h,i,j] = how much token i attends to token j in head h of batch b 
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        #softmax along the last dimension (sum to 1 for attention weights)
        attn_weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, v)

        #return the shape to the original
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(batch_size, seq_len, self.d_model)

        #final output
        output = self.out_linear(context)

        return output 


# --- Feed-Forward Network (FFN) ---
class FeedForward(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.d_model = config.d_model
        self.d_ff = config.d_model * config.ffn_mult

        self.linear1 = nn.Linear(self.d_model, self.d_ff)
        self.linear2 = nn.Linear(self.d_ff, self.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        x = self.dropout(x)
        return x


# --- Transformer Block ---
class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.attn = MultiHeadSelfAttention(config)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.ffn = FeedForward(config)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, mask=None):
        attn_out = self.attn(x)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        out = self.norm2(x + ffn_out)
        return out 


# --- Transformer Encoder Stack ---
class TransformerEncoder(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.embed = TokenPositionalEmbedding(config)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, x, mask=None):
        # 1. Embed tokens (add position info)
        x = self.embed(x)
        
        # 2. Pass through each TransformerBlock
        for layer in self.layers:
            x = layer(x)
        
        # 3. Apply final normalization
        x = self.norm(x)

        return x

# --- Output Head ---
# final linear projection from d_model to vocab_size 
class OutputHead(nn.Module):
    def __init__(self, config: TransformerConfig, embedding_layer):
        super().__init__()
        self.proj = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, x):
        # Input: (batch, seq_len, d_model)
        # Output: (batch, seq_len, vocab_size)
        logits = self.proj(x)
        return logits

# --- Full Model ---
class MinimalTransformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.encoder = TransformerEncoder(config)
        self.head = OutputHead(config, self.encoder.embed)


    def forward(self, token_ids, mask=None):
        hidden = self.encoder(token_ids)       # (batch, seq_len, d_model)
        logits = self.head(hidden)     # (batch, seq_len, vocab_size)
        return logits
