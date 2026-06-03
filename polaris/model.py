"""GPT-style transformer for `polaris`.

Defines the model and its block. The block is intentionally structured so
that its feed-forward sub-layer is an **FFN object satisfying a common
interface**, not a hard-coded MLP. Both `polaris.ffn.dense` and
`polaris.ffn.moe` implement that interface, so swapping arms of the study
is a config change (`ffn_type: dense | moe`) — not a code change — and a
paired (dense, moe) run differs *only* in the FFN module.

The FFN interface, in prose (you write the actual signature in `ffn/`):

- It is an `nn.Module`.
- Forward takes a token-stream tensor of shape (B, T, C) and returns a
  tensor of the same shape.
- For MoE it additionally exposes whatever auxiliary signals the training
  loop needs to add the load-balancing loss and to record per-expert load
  (e.g. as a second return value, or as an attribute set during forward).
  Decide which discipline you want and apply it consistently across both
  FFN types so the training loop is uniform.

Borrowing note: the attention (causal self-attention with a triangular
mask), the LayerNorm/RMSNorm choice, and the residual block wiring are the
intellectually load-bearing parts of the model — write them yourself. The
token + position embeddings and the LM head are pure plumbing and are fair
game for direct borrowing from nanoGPT (mark them as such in CREDITS.md).
"""

# TODO(human): heads-up — the project is on ROCm 5.7 specifically
#              because ROCm 6.x has a gfx803 fp32 GEMM bug. Always run
#              `python env/check-matmul.py --dtype fp32` before training
#              to confirm your stack hasn't drifted. See KNOWN_GOTCHAS.md.
# TODO(human): implement the block:
#              norm -> attention -> residual -> norm -> FFN(via interface)
#              -> residual.
# TODO(human): implement causal self-attention (multi-head, fp32, no flash).
# TODO(human): wire token + position embeddings and the LM head
#              (may be borrowed verbatim from nanoGPT — note in CREDITS).
# TODO(human): instantiate the FFN in each block by looking up
#              `config.ffn_type` and dispatching to the dense or MoE class.
# TODO(human): expose a `from_config(cfg)` (or similar) that builds the whole
#              model given a Config and returns it ready to `.to(device)`.

import torch
import torch.nn as nn

from .ffn.dense import DenseLayer


class MultiHeadAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.d_head = cfg.d_head
        self.n_head = cfg.n_head
        self.W_Q = nn.Linear(cfg.d_model, self.n_head * self.d_head, bias=cfg.qkv_bias)
        self.W_K = nn.Linear(cfg.d_model, self.n_head * self.d_head, bias=cfg.qkv_bias)
        self.W_V = nn.Linear(cfg.d_model, self.n_head * self.d_head, bias=cfg.qkv_bias)
        self.W_O = nn.Linear(self.n_head * self.d_head, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(cfg.max_seq_len, cfg.max_seq_len), diagonal=1).bool(),
        )

    def forward(self, x):
        b, n_tokens, _d_model = x.shape

        keys = self.W_K(x)
        queries = self.W_Q(x)
        values = self.W_V(x)

        keys = keys.view(b, n_tokens, self.n_head, self.d_head)
        queries = queries.view(b, n_tokens, self.n_head, self.d_head)
        values = values.view(b, n_tokens, self.n_head, self.d_head)

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        mask = self.mask[:n_tokens, :n_tokens]

        attn_scores = attn_scores.masked_fill(mask, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(
            b, n_tokens, self.n_head * self.d_head
        )
        context_vec = self.W_O(context_vec)

        return context_vec


class LayerNorm(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(d_model))
        self.shift = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(cfg)
        if cfg.ffn_type == "dense":
            self.ffn = DenseLayer(cfg)
        elif cfg.ffn_type == "moe":
            raise NotImplementedError
        else:
            raise ValueError(f"Unknown ffn_type: {cfg.ffn_type}")
        self.norm1 = LayerNorm(cfg.d_model)
        self.norm2 = LayerNorm(cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.dropout(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x, aux = self.ffn(x)
        x = self.dropout(x)
        x = x + shortcut

        return x, aux


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop_emb = nn.Dropout(cfg.dropout)

        self.trf_blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg.n_layer)]
        )

        self.final_norm = LayerNorm(cfg.d_model)
        self.out_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, x):
        _batch_size, seq_len = x.size()

        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(torch.arange(seq_len, device=x.device))

        x = self.drop_emb(tok_emb + pos_emb)

        aux_losses = []
        for block in self.trf_blocks:
            x, aux = block(x)
            aux_losses.append(aux)

        x = self.final_norm(x)
        logits = self.out_head(x)

        return logits, aux_losses
