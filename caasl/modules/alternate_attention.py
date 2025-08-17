from torch import nn


class AlternateAttention(nn.Module):
    """
    Alternate Attention module.

    Args:
        dim (int): Dimension of the input.
        feedforward_dim (int): Dimension of the feedforward layer.
        dropout (float): Dropout rate.
        n_layers (int): Number of layers.
        num_heads (int): Number of attention heads.
    """

    def __init__(self, dim, feedforward_dim, *, dropout=0.1, n_layers=2, num_heads=8):
        super().__init__()
        self._dim = dim
        self._feedforward_dim = feedforward_dim
        self._dropout = dropout
        self._n_layers = 2 * n_layers  # Alternate Attention
        self._num_heads = num_heads

        self.proj = nn.Linear(2, self._dim)
        self.attentions = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=self._dim,
                    nhead=self._num_heads,
                    dropout=self._dropout,
                    dim_feedforward=self._feedforward_dim,
                    norm_first=True,
                    batch_first=True,
                )
                for _ in range(self._n_layers)
            ]
        )
        self._layernorm = nn.LayerNorm(dim)

    def forward(self, x):
        z = self.proj(x)
        for i in range(self._n_layers):
            *leading_dims, d1, d2, n_features = z.shape
            z = z.reshape(-1, d2, n_features)
            z = self.attentions[i](z)
            z = z.reshape(*leading_dims, d1, d2, n_features)
            z = z.transpose(-2, -3)
        z = self._layernorm(z)
        return z
