import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from typing import Tuple


# if concatenate or splitted (change)
# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1024, combined=False):
        super(PositionalEncoding, self).__init__()
        if combined:
            view_pe = nn.Parameter(torch.randn(1, max_len, d_model))
            self.pos_embedding = torch.cat([view_pe, view_pe], dim=1)
        else:
            self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model))

    def forward(self, x):
        device = x.device
        self.pos_embedding = self.pos_embedding.to(device)
        return x + self.pos_embedding


def MLP_PE(channels: list, do_bn: bool = True) -> nn.Module:
    """Multi-layer perceptron"""
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n - 1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


# Rotary encoding
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def reshape_for_broadcast(freqs_cis, x: Tensor) -> Tensor:
    return freqs_cis.unsqueeze(0).unsqueeze(0).expand(*x.shape[:-1], -1)


def apply_rotary_emb(
    xq: Tensor, xk: Tensor, freqs_cis: Tensor
) -> Tuple[Tensor, Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Normalization(nn.Module):
    def __init__(self, norm_type, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.dim = dim
        self.norm_type = norm_type

        if norm_type == "layer":
            self.norm = nn.LayerNorm(dim)
        elif norm_type == "batch":
            self.norm = nn.BatchNorm1d(dim)
        else:
            raise ValueError(f"Unknown normalization type: {norm_type}")

    def forward(self, x):
        return self.norm(x)

# MLP
class MLP(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        bias=True,
        activation="gelu",
        norm_type="layer",
        layer_norm_input=False,
        layer_norm_output=True,
    ):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=bias)
        self.norm = Normalization(norm_type, input_dim)
        self.norm2 = Normalization(norm_type, output_dim)
        self.layer_norm_input = layer_norm_input
        self.layer_norm_output = layer_norm_output
        self.activation = activation

    def forward(self, x):
        if self.layer_norm_input:
            x = self.norm(x)
        if self.activation == "gelu":
            x = F.gelu(self.fc1(x))
        elif self.activation == "relu":
            x = F.relu(self.fc1(x))
        else:
            raise ValueError(f"Unknown activation function: {self.activation}")

        x = self.fc2(x)
        if self.layer_norm_output:
            x = self.norm2(x)
        return x


# Attention Layer
class AttentionLayer(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads=8,
        dropout=0.1,
        pos_encoding_name=None,
        pos_encoding=None,
        bias=True,
        norm_type="layer",
        layer_norm_input=True,
        layer_norm_output=True,
    ):
        super().__init__()
        dim_head = d_model // num_heads
        self.heads = num_heads
        self.scale = dim_head**-0.5

        self.layer_norm_input = layer_norm_input
        if self.layer_norm_input:
            self.norm_q = Normalization(norm_type, d_model)
            self.norm_k = Normalization(norm_type, d_model)
            self.norm_v = Normalization(norm_type, d_model)

        self.to_q = nn.Linear(d_model, d_model, bias=bias)
        self.to_k = nn.Linear(d_model, d_model, bias=bias)
        self.to_v = nn.Linear(d_model, d_model, bias=bias)

        self.attend = nn.Softmax(dim=-1)
        self.to_out = nn.Sequential(
            nn.Linear(d_model, d_model, bias=False),
            nn.Dropout(dropout),
        )

        self.layer_norm_output = layer_norm_output
        if self.layer_norm_output:
            self.layer_norm = Normalization(norm_type, d_model)

        self.pos_enc_name = pos_encoding_name
        if pos_encoding_name == "learnable":
            self.positional_encoding = pos_encoding
        elif pos_encoding_name == "rotary":
            self.freqs_cis = precompute_freqs_cis(dim_head, 1024)
            self.positional_encoding = None
        else:
            self.positional_encoding = None

    def forward(self, x, source, mask=None):
        if self.pos_enc_name == "learnable":
            x = self.positional_encoding(x)
            source = self.positional_encoding(source)

        if self.layer_norm_input:
            q = self.to_q(self.norm_q(source))
            k = self.to_k(self.norm_k(x))
            v = self.to_v(self.norm_v(x))

        else:
            q = self.to_q(source)
            k = self.to_k(x)
            v = self.to_v(x)

        q, k, v = (
            rearrange(q, "b n (h d) -> b h n d", h=self.heads),
            rearrange(k, "b n (h d) -> b h n d", h=self.heads),
            rearrange(v, "b n (h d) -> b h n d", h=self.heads),
        )
        
        if self.pos_enc_name == "rotary":
            bsz, seqlen, _ = x.size()
            freqs_cis = self.freqs_cis[:seqlen].to(x.device)
            q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        if mask is not None:
            dots = dots.masked_fill(mask == 0, float("-inf"))

        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        if self.layer_norm_output:
            out = self.layer_norm(out)
        return out


class Layer(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        hidden_dim,
        dropout=0.1,
        pos_encoding="learnable",
        transformer_norm_type="layer",
        transformer_split_self_cross=False,
        transformer_parallel_self_cross=False,
        transformer_message_pass=False,
        transformer_mlp_input="last",
        transformer_mlp_bias=True,
        transformer_mlp_activation="gelu",
        mlp_layer_norm_input=False,
        mlp_layer_norm_output=True,
        transformer_attention_bias=True,
        attention_layer_norm_input=True,
        attention_layer_norm_output=True,
        norm_after_add=True,
    ):
        super(Layer, self).__init__()
        self.transformer_split_self_cross = transformer_split_self_cross
        self.transformer_parallel_self_cross = transformer_parallel_self_cross
        self.transformer_message_pass = transformer_message_pass
        self.transformer_mlp_input = transformer_mlp_input
        self.dropout = nn.Dropout(dropout)

        if pos_encoding == "learnable":
            if self.transformer_split_self_cross:
                self.positional_encoding = PositionalEncoding(
                    d_model, max_len=256, combined=False
                )
            else:
                self.positional_encoding = PositionalEncoding(
                    d_model, max_len=256, combined=True
                )
        else:
            self.positional_encoding = None

        # all models
        self.self_attention_layer = AttentionLayer(
            d_model,
            num_heads,
            dropout,
            pos_encoding_name=pos_encoding,
            pos_encoding=self.positional_encoding,
            bias=transformer_attention_bias,
            norm_type=transformer_norm_type,
            layer_norm_input=attention_layer_norm_input,
            layer_norm_output=attention_layer_norm_output,
        )

        if self.transformer_message_pass:
            self.mlp_self = MLP(
                d_model * 2,
                hidden_dim,
                d_model,
                bias=transformer_mlp_bias,
                activation=transformer_mlp_activation,
                norm_type=transformer_norm_type,
                layer_norm_input=mlp_layer_norm_input,
                layer_norm_output=mlp_layer_norm_output,
            )
        else:
            self.mlp_self = MLP(
                d_model,
                hidden_dim,
                d_model,
                bias=transformer_mlp_bias,
                activation=transformer_mlp_activation,
                norm_type=transformer_norm_type,
                layer_norm_input=mlp_layer_norm_input,
                layer_norm_output=mlp_layer_norm_output,
            )

        # self-cross / loftr / superglue ...
        if self.transformer_split_self_cross:
            self.cross_attention_layer = AttentionLayer(
                d_model,
                num_heads,
                dropout,
                pos_encoding_name=None,
                pos_encoding=None,
                bias=transformer_attention_bias,
                norm_type=transformer_norm_type,
                layer_norm_input=attention_layer_norm_input,
                layer_norm_output=attention_layer_norm_output,
            )
            if self.transformer_message_pass:
                self.mlp_feature_delta = MLP(
                    d_model * 2,
                    hidden_dim,
                    d_model,
                    bias=transformer_mlp_bias,
                    activation=transformer_mlp_activation,
                    norm_type=transformer_norm_type,
                    layer_norm_input=mlp_layer_norm_input,
                    layer_norm_output=mlp_layer_norm_output,
                )
            else:
                if self.transformer_parallel_self_cross:
                    self.mlp_feature_delta = MLP(
                        d_model * 2,
                        hidden_dim,
                        d_model,
                        bias=transformer_mlp_bias,
                        activation=transformer_mlp_activation,
                        norm_type=transformer_norm_type,
                        layer_norm_input=mlp_layer_norm_input,
                        layer_norm_output=mlp_layer_norm_output,
                    )
                else:
                    self.mlp_feature_delta = MLP(
                        d_model,
                        hidden_dim,
                        d_model,
                        bias=transformer_mlp_bias,
                        activation=transformer_mlp_activation,
                        norm_type=transformer_norm_type,
                        layer_norm_input=mlp_layer_norm_input,
                        layer_norm_output=mlp_layer_norm_output,
                    )

            # self-cross
            if self.transformer_parallel_self_cross:
                if self.transformer_message_pass:
                    self.mlp_cross = MLP(
                        d_model * 2,
                        hidden_dim,
                        d_model,
                        bias=transformer_mlp_bias,
                        activation=transformer_mlp_activation,
                        norm_type=transformer_norm_type,
                        layer_norm_input=mlp_layer_norm_input,
                        layer_norm_output=mlp_layer_norm_output,
                    )
                else:
                    self.mlp_cross = MLP(
                        d_model,
                        hidden_dim,
                        d_model,
                        bias=transformer_mlp_bias,
                        activation=transformer_mlp_activation,
                        norm_type=transformer_norm_type,
                        layer_norm_input=mlp_layer_norm_input,
                        layer_norm_output=mlp_layer_norm_output,
                    )

        self.norm_after_add = norm_after_add
        if self.norm_after_add:
            self.layer_norm_1_self = Normalization(transformer_norm_type, d_model)
            self.layer_norm_2_self = Normalization(transformer_norm_type, d_model)
            if self.transformer_split_self_cross:
                self.layer_norm_1_cross = Normalization(transformer_norm_type, d_model)
                self.layer_norm_2_cross = Normalization(transformer_norm_type, d_model)
                if self.transformer_parallel_self_cross:
                    self.layer_norm_1 = Normalization(transformer_norm_type, d_model)
                    self.layer_norm_2 = Normalization(transformer_norm_type, d_model)

    def forward(self, features_1, features_2, mask=None):

        # vit / combined
        if not self.transformer_split_self_cross:
            features = torch.cat([features_1, features_2], dim=1)
            attn_output = self.self_attention_layer(features, features)  # no mask

            # vit
            if self.transformer_message_pass:
                message = self.mlp_self(torch.cat([features, attn_output], dim=-1))
                features = features + message

            # combined features message passing
            else:
                combined_features = attn_output + features
                message = self.mlp_self(combined_features)
                if (
                    self.transformer_mlp_input == "original"
                ):  # add with initial features
                    features = features + message
                else:  # add with self-attention output
                    features = combined_features + message

            if self.norm_after_add:
                features = self.layer_norm_1_self(features)

            features_1, features_2 = torch.split(
                features, [features_1.size(1), features_2.size(1)], dim=1
            )

        else:
            # Self-attention
            delta_self_1 = self.self_attention_layer(features_1, features_1)
            delta_self_2 = self.self_attention_layer(features_2, features_2)

            if self.transformer_parallel_self_cross:
                # Cross-attention
                delta_cross_12 = self.cross_attention_layer(
                    features_1, features_2, mask
                )
                delta_cross_21 = self.cross_attention_layer(
                    features_2, features_1, mask
                )
                if self.transformer_message_pass:  # concat_mlp and add
                    message_1_self = self.mlp_self(
                        torch.cat([features_1, delta_self_1], dim=-1)
                    )
                    message_2_self = self.mlp_self(
                        torch.cat([features_2, delta_self_2], dim=-1)
                    )
                    features_1_self = features_1 + message_1_self
                    features_2_self = features_2 + message_2_self

                    if self.norm_after_add:
                        features_1_self = self.layer_norm_1_self(features_1_self)
                        features_2_self = self.layer_norm_2_self(features_2_self)
                    # Cross-attention update with residual connection and Normalization
                    message_1_cross = self.mlp_cross(
                        torch.cat([features_1, delta_cross_21], dim=-1)
                    )
                    message_2_cross = self.mlp_cross(
                        torch.cat([features_2, delta_cross_12], dim=-1)
                    )
                    features_1_cross = features_1 + message_1_cross
                    features_2_cross = features_2 + message_2_cross

                    if self.norm_after_add:
                        features_1_cross = self.layer_norm_1_cross(features_1_cross)
                        features_2_cross = self.layer_norm_2_cross(features_2_cross)

                    # Feature delta update with residual connection and Normalization
                    message_1 = self.mlp_feature_delta(
                        torch.cat([features_1_self, features_1_cross], dim=-1)
                    )
                    message_2 = self.mlp_feature_delta(
                        torch.cat([features_2_self, features_2_cross], dim=-1)
                    )
                    features_1 = features_1 + message_1
                    features_2 = features_2 + message_2
                    if self.norm_after_add:
                        features_1 = self.layer_norm_1(features_1)
                        features_2 = self.layer_norm_2(features_2)

                else:
                    features_1_self = features_1 + delta_self_1
                    features_2_self = features_2 + delta_self_2
                    if self.transformer_mlp_input == "original":
                        features_1_self = features_1 + self.mlp_self(features_1_self)
                        features_2_self = features_2 + self.mlp_self(features_2_self)
                    else:
                        features_1_self = features_1_self + self.mlp_self(
                            features_1_self
                        )
                        features_2_self = features_2_self + self.mlp_self(
                            features_2_self
                        )
                    if self.norm_after_add:
                        features_1_self = self.layer_norm_1_self(features_1_self)
                        features_2_self = self.layer_norm_2_self(features_2_self)

                    features_1_cross = features_1 + delta_cross_21
                    features_2_cross = features_2 + delta_cross_12
                    if self.transformer_mlp_input == "original":
                        features_1_cross = features_1 + self.mlp_cross(features_1_cross)
                        features_2_cross = features_2 + self.mlp_cross(features_2_cross)
                    else:
                        features_1_cross = features_1_cross + self.mlp_cross(
                            features_1_cross
                        )
                        features_2_cross = features_2_cross + self.mlp_cross(
                            features_2_cross
                        )
                    if self.norm_after_add:
                        features_1_cross = self.layer_norm_1_cross(features_1_cross)
                        features_2_cross = self.layer_norm_2_cross(features_2_cross)

                    message_1 = self.mlp_feature_delta(
                        torch.cat([features_1_self, features_1_cross], dim=-1)
                    )
                    message_2 = self.mlp_feature_delta(
                        torch.cat([features_2_self, features_2_cross], dim=-1)
                    )
                    features_1 = features_1 + message_1
                    features_2 = features_2 + message_2
                    if self.norm_after_add:
                        features_1 = self.layer_norm_1(features_1)
                        features_2 = self.layer_norm_2(features_2)

            else:
                # cascaded MP
                if self.transformer_message_pass:
                    message_1 = self.mlp_feature_delta(
                        torch.cat([features_1, delta_self_1], dim=-1)
                    )
                    message_2 = self.mlp_feature_delta(
                        torch.cat([features_2, delta_self_2], dim=-1)
                    )
                    features_1 = features_1 + message_1
                    features_2 = features_2 + message_2
                    if self.norm_after_add:
                        features_1 = self.layer_norm_1_self(features_1)
                        features_2 = self.layer_norm_2_self(features_2)
                    delta_cross_12 = self.cross_attention_layer(
                        features_1, features_2, mask
                    )
                    delta_cross_21 = self.cross_attention_layer(
                        features_2, features_1, mask
                    )
                    message_1 = self.mlp_feature_delta(
                        torch.cat([features_1, delta_cross_21], dim=-1)
                    )
                    message_2 = self.mlp_feature_delta(
                        torch.cat([features_2, delta_cross_12], dim=-1)
                    )
                    features_1 = features_1 + message_1
                    features_2 = features_2 + message_2
                    if self.norm_after_add:
                        features_1 = self.layer_norm_1_cross(features_1)
                        features_2 = self.layer_norm_2_cross(features_2)
                else:
                    # Cascaded self/cross vit
                    features_1_self = features_1 + delta_self_1
                    features_2_self = features_2 + delta_self_2
                    if self.transformer_mlp_input == "original":
                        features_1 = features_1 + self.mlp_self(features_1_self)
                        features_2 = features_2 + self.mlp_self(features_2_self)
                    else:
                        features_1 = features_1_self + self.mlp_self(features_1_self)
                        features_2 = features_2_self + self.mlp_self(features_2_self)
                    if self.norm_after_add:
                        features_1 = self.layer_norm_1_self(features_1)
                        features_2 = self.layer_norm_2_self(features_2)

                    delta_cross_21 = self.cross_attention_layer(
                        features_2, features_1, mask
                    )
                    delta_cross_12 = self.cross_attention_layer(
                        features_1, features_2, mask
                    )
                    features_1_cross = features_1 + delta_cross_21
                    features_2_cross = features_2 + delta_cross_12
                    if self.transformer_mlp_input == "original":
                        features_1 = features_1 + self.mlp_feature_delta(
                            features_1_cross
                        )
                        features_2 = features_2 + self.mlp_feature_delta(
                            features_2_cross
                        )
                    else:
                        features_1 = features_1_cross + self.mlp_feature_delta(
                            features_1_cross
                        )
                        features_2 = features_2_cross + self.mlp_feature_delta(
                            features_2_cross
                        )
                    if self.norm_after_add:
                        features_1 = self.layer_norm_1_cross(features_1)
                        features_2 = self.layer_norm_2_cross(features_2)

        return features_1, features_2


class Layer_only_self(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        hidden_dim,
        dropout=0.1,
        pos_encoding="learnable",
        transformer_norm_type="layer",
        transformer_split_self_cross=False,
        transformer_parallel_self_cross=False,
        transformer_message_pass=False,
        transformer_mlp_input="last",
        transformer_mlp_bias=True,
        transformer_mlp_activation="gelu",
        mlp_layer_norm_input=False,
        mlp_layer_norm_output=True,
        transformer_attention_bias=True,
        attention_layer_norm_input=True,
        attention_layer_norm_output=True,
        norm_after_add=True,
    ):
        super(Layer_only_self, self).__init__()
        self.transformer_split_self_cross = transformer_split_self_cross
        self.transformer_parallel_self_cross = transformer_parallel_self_cross
        self.transformer_message_pass = transformer_message_pass
        self.transformer_mlp_input = transformer_mlp_input
        self.dropout = nn.Dropout(dropout)

        if pos_encoding == "learnable":
            if self.transformer_split_self_cross:
                self.positional_encoding = PositionalEncoding(d_model, max_len=256)
            else:
                self.positional_encoding = PositionalEncoding(d_model, max_len=512)
        else:
            self.positional_encoding = None

        # all models
        self.self_attention_layer = AttentionLayer(
            d_model,
            num_heads,
            dropout,
            pos_encoding_name=pos_encoding,
            pos_encoding=self.positional_encoding,
            bias=transformer_attention_bias,
            norm_type=transformer_norm_type,
            layer_norm_input=attention_layer_norm_input,
            layer_norm_output=attention_layer_norm_output,
        )

        if self.transformer_message_pass:
            self.mlp_self = MLP(
                d_model * 2,
                hidden_dim,
                d_model,
                bias=transformer_mlp_bias,
                activation=transformer_mlp_activation,
                norm_type=transformer_norm_type,
                layer_norm_input=mlp_layer_norm_input,
                layer_norm_output=mlp_layer_norm_output,
            )
        else:
            self.mlp_self = MLP(
                d_model,
                hidden_dim,
                d_model,
                bias=transformer_mlp_bias,
                activation=transformer_mlp_activation,
                norm_type=transformer_norm_type,
                layer_norm_input=mlp_layer_norm_input,
                layer_norm_output=mlp_layer_norm_output,
            )

        self.norm_after_add = norm_after_add
        if self.norm_after_add:
            self.layer_norm_1_self = Normalization(transformer_norm_type, d_model)
            self.layer_norm_2_self = Normalization(transformer_norm_type, d_model)

    def forward(self, features_1, features_2, mask=None):
        if self.transformer_split_self_cross:
            delta_self_1 = self.self_attention_layer(features_1, features_1)
            delta_self_2 = self.self_attention_layer(features_2, features_2)
            if self.transformer_message_pass:
                message_1 = self.mlp_self(torch.cat([features_1, delta_self_1], dim=-1))
                message_2 = self.mlp_self(torch.cat([features_2, delta_self_2], dim=-1))
                features_1 = features_1 + message_1
                features_2 = features_2 + message_2
                if self.norm_after_add:
                    features_1 = self.layer_norm_1_self(features_1)
                    features_2 = self.layer_norm_2_self(features_2)
            else:
                features_1 = features_1 + delta_self_1
                features_2 = features_2 + delta_self_2
                if self.transformer_mlp_input == "original":
                    features_1 = features_1 + self.mlp_self(features_1)
                    features_2 = features_2 + self.mlp_self(features_2)
                else:
                    features_1 = features_1 + self.mlp_self(features_1)
                    features_2 = features_2 + self.mlp_self(features_2)
                if self.norm_after_add:
                    features_1 = self.layer_norm_1_self(features_1)
                    features_2 = self.layer_norm_2_self(features_2)
        else:
            NotImplementedError("Not implemented yet")
        return features_1, features_2


class Layer_only_cross(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        hidden_dim,
        dropout=0.1,
        pos_encoding="learnable",
        transformer_norm_type="layer",
        transformer_split_self_cross=False,
        transformer_parallel_self_cross=False,
        transformer_message_pass=False,
        transformer_mlp_input="last",
        transformer_mlp_bias=True,
        transformer_mlp_activation="gelu",
        mlp_layer_norm_input=False,
        mlp_layer_norm_output=True,
        transformer_attention_bias=True,
        attention_layer_norm_input=True,
        attention_layer_norm_output=True,
        norm_after_add=True,
    ):
        super(Layer_only_cross, self).__init__()
        self.transformer_split_self_cross = transformer_split_self_cross
        self.transformer_parallel_self_cross = transformer_parallel_self_cross
        self.transformer_message_pass = transformer_message_pass
        self.transformer_mlp_input = transformer_mlp_input
        self.dropout = nn.Dropout(dropout)

        self.cross_attention_layer = AttentionLayer(
            d_model,
            num_heads,
            dropout,
            pos_encoding_name=None,
            pos_encoding=None,
            bias=transformer_attention_bias,
            norm_type=transformer_norm_type,
            layer_norm_input=attention_layer_norm_input,
            layer_norm_output=attention_layer_norm_output,
        )
        if self.transformer_message_pass:
            self.mlp_feature_delta = MLP(
                d_model * 2,
                hidden_dim,
                d_model,
                bias=transformer_mlp_bias,
                activation=transformer_mlp_activation,
                norm_type=transformer_norm_type,
                layer_norm_input=mlp_layer_norm_input,
                layer_norm_output=mlp_layer_norm_output,
            )
        else:
            self.mlp_feature_delta = MLP(
                d_model,
                hidden_dim,
                d_model,
                bias=transformer_mlp_bias,
                activation=transformer_mlp_activation,
                norm_type=transformer_norm_type,
                layer_norm_input=mlp_layer_norm_input,
                layer_norm_output=mlp_layer_norm_output,
            )

        self.norm_after_add = norm_after_add
        if self.norm_after_add:
            self.layer_norm_1_cross = Normalization(transformer_norm_type, d_model)
            self.layer_norm_2_cross = Normalization(transformer_norm_type, d_model)

    def forward(self, features_1, features_2, mask=None):
        if self.transformer_split_self_cross:
            if self.transformer_message_pass:
                delta_cross_12 = self.cross_attention_layer(
                    features_1, features_2, mask
                )
                delta_cross_21 = self.cross_attention_layer(
                    features_2, features_1, mask
                )
                message_1 = self.mlp_feature_delta(
                    torch.cat([features_1, delta_cross_21], dim=-1)
                )
                message_2 = self.mlp_feature_delta(
                    torch.cat([features_2, delta_cross_12], dim=-1)
                )
                features_1 = features_1 + message_1
                features_2 = features_2 + message_2
                if self.norm_after_add:
                    features_1 = self.layer_norm_1_cross(features_1)
                    features_2 = self.layer_norm_2_cross(features_2)
            else:
                delta_cross_21 = self.cross_attention_layer(
                    features_2, features_1, mask
                )
                delta_cross_12 = self.cross_attention_layer(
                    features_1, features_2, mask
                )
                features_1_cross = features_1 + delta_cross_21
                features_2_cross = features_2 + delta_cross_12
                if self.transformer_mlp_input == "original":
                    features_1 = features_1 + self.mlp_feature_delta(features_1_cross)
                    features_2 = features_2 + self.mlp_feature_delta(features_2_cross)
                else:
                    features_1 = features_1_cross + self.mlp_feature_delta(
                        features_1_cross
                    )
                    features_2 = features_2_cross + self.mlp_feature_delta(
                        features_2_cross
                    )
                if self.norm_after_add:
                    features_1 = self.layer_norm_1_cross(features_1)
                    features_2 = self.layer_norm_2_cross(features_2)
        else:
            NotImplementedError("Not implemented yet")
        return features_1, features_2


# Transformer
class Transformer(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        hidden_dim,
        num_layers,
        dropout=0.1,
        pos_encoding="learnable",
        transformer_layers="self_cross_alternate",
        transformer_norm_type="layer",
        transformer_split_self_cross=False,
        transformer_parallel_self_cross=False,
        transformer_message_pass=False,
        transformer_mlp_input="last",
        transformer_final_projection=False,
        transformer_mlp_bias=True,
        transformer_mlp_activation="gelu",
        mlp_layer_norm_input=False,
        mlp_layer_norm_output=True,
        transformer_attention_bias=True,
        attention_layer_norm_input=True,
        attention_layer_norm_output=True,
        norm_after_add=True,
    ):
        super(Transformer, self).__init__()

        if transformer_split_self_cross and transformer_layers == "self_cross_seperate":
            self.self_layers = nn.ModuleList(
                [
                    Layer_only_self(
                        d_model,
                        num_heads,
                        hidden_dim,
                        dropout,
                        pos_encoding,
                        transformer_norm_type,
                        transformer_split_self_cross,
                        transformer_parallel_self_cross,
                        transformer_message_pass,
                        transformer_mlp_input,
                        transformer_mlp_bias,
                        transformer_mlp_activation,
                        mlp_layer_norm_input,
                        mlp_layer_norm_output,
                        transformer_attention_bias,
                        attention_layer_norm_input,
                        attention_layer_norm_output,
                        norm_after_add,
                    )
                    for _ in range(num_layers)
                ]
            )
            self.cross_layers = nn.ModuleList(
                [
                    Layer_only_cross(
                        d_model,
                        num_heads,
                        hidden_dim,
                        dropout,
                        pos_encoding,
                        transformer_norm_type,
                        transformer_split_self_cross,
                        transformer_parallel_self_cross,
                        transformer_message_pass,
                        transformer_mlp_input,
                        transformer_mlp_bias,
                        transformer_mlp_activation,
                        mlp_layer_norm_input,
                        mlp_layer_norm_output,
                        transformer_attention_bias,
                        attention_layer_norm_input,
                        attention_layer_norm_output,
                        norm_after_add,
                    )
                    for _ in range(num_layers)
                ]
            )
        # if transformer_layers == "self_cross_alternate":
        else:
            self.layers = nn.ModuleList(
                [
                    Layer(
                        d_model,
                        num_heads,
                        hidden_dim,
                        dropout,
                        pos_encoding,
                        transformer_norm_type,
                        transformer_split_self_cross,
                        transformer_parallel_self_cross,
                        transformer_message_pass,
                        transformer_mlp_input,
                        transformer_mlp_bias,
                        transformer_mlp_activation,
                        mlp_layer_norm_input,
                        mlp_layer_norm_output,
                        transformer_attention_bias,
                        attention_layer_norm_input,
                        attention_layer_norm_output,
                        norm_after_add,
                    )
                    for _ in range(num_layers)
                ]
            )

        self.d_model = d_model
        self.transformer_layers = transformer_layers
        self.transformer_split_self_cross = transformer_split_self_cross
        self.transformer_final_projection = transformer_final_projection
        if self.transformer_final_projection:
            self.final_projection = nn.Conv1d(
                d_model, d_model, kernel_size=1, bias=True
            )
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, features_1, features_2, mask=None):
        bs = features_1.size(0)
        # features_1, features_2 = features_1.contiguous().view(
        #     bs, -1, self.d_model
        # ), features_2.contiguous().view(bs, -1, self.d_model)

        if self.transformer_split_self_cross:
            if self.transformer_layers == "self_cross_seperate":
                for layer in self.self_layers:
                    features_1, features_2 = layer(
                        features_1,
                        features_2,  # mask
                    )
                for layer in self.cross_layers:
                    features_1, features_2 = layer(
                        features_1,
                        features_2,  # mask
                    )
        else:  # self_cross_alternate
            for layer in self.layers:
                features_1, features_2 = layer(
                    features_1,
                    features_2,  # mask
                )

            # calculate correspondence matrix / confidence matrix_score / loss , return if reached desired loss - early stopping
        if self.transformer_final_projection:
            features_1 = self.final_projection(features_1)
            features_2 = self.final_projection(features_2)
        return features_1, features_2


# Transformer Builder
class TransformerBuilder:
    def __init__(self):
        self.d_model = 512
        self.num_heads = 8
        self.hidden_dim = 512
        self.num_layers = 6
        self.dropout = 0.1
        self.pos_encoding = "learnable"
        self.transformer_layers = "self_cross_alternate"  # self_cross_seperate
        self.transformer_norm_type = "layer"
        self.transformer_split_self_cross = False
        self.transformer_parallel_self_cross = False
        self.transformer_message_pass = False
        self.transformer_mlp_input = "last"  # original or last
        self.transformer_final_projection = False
        self.transformer_mlp_bias = True
        self.transformer_mlp_activation = "gelu"
        self.mlp_layer_norm_input = False
        self.mlp_layer_norm_output = True
        self.transformer_attention_bias = True
        self.attention_layer_norm_input = True
        self.attention_layer_norm_output = True
        self.norm_after_add = True

    def set_d_model(self, d_model):
        self.d_model = d_model
        return self

    def set_num_heads(self, num_heads):
        self.num_heads = num_heads
        return self

    def set_hidden_dim(self, hidden_dim):
        self.hidden_dim = hidden_dim
        return self

    def set_num_layers(self, num_layers):
        self.num_layers = num_layers
        return self

    def set_dropout(self, dropout):
        self.dropout = dropout
        return self

    def set_pos_encoding(self, pos_encoding):
        self.pos_encoding = pos_encoding
        return self

    def set_transformer_layers(self, transformer_layers):
        self.transformer_layers = transformer_layers
        return self

    def set_transformer_type(self, transformer_type):
        self.transformer_type = transformer_type
        return self

    def set_transformer_norm_type(self, transformer_norm_type):
        self.transformer_norm_type = transformer_norm_type
        return self

    def set_transformer_split_self_cross(self, transformer_split_self_cross):
        self.transformer_split_self_cross = transformer_split_self_cross
        return self

    def set_transformer_parallel_self_cross(self, transformer_parallel_self_cross):
        self.transformer_parallel_self_cross = transformer_parallel_self_cross
        return self

    def set_transformer_message_pass(self, transformer_message_pass):
        self.transformer_message_pass = transformer_message_pass
        return self

    def set_transformer_mlp_input(self, transformer_mlp_input):
        self.transformer_mlp_input = transformer_mlp_input
        return self

    def set_transformer_final_projection(self, transformer_final_projection):
        self.transformer_final_projection = transformer_final_projection
        return

    def set_transformer_attention_bias(self, transformer_attention_bias):
        self.transformer_attention_bias = transformer_attention_bias
        return self

    def set_transformer_mlp_bias(self, transformer_mlp_bias):
        self.transformer_mlp_bias = transformer_mlp_bias
        return self

    def set_transformer_mlp_activation(self, transformer_mlp_activation):
        self.transformer_mlp_activation = transformer_mlp_activation
        return self

    def set_mlp_layer_norm_input(self, mlp_layer_norm_input):
        self.mlp_layer_norm_input = mlp_layer_norm_input
        return self

    def set_mlp_layer_norm_output(self, mlp_layer_norm_output):
        self.mlp_layer_norm_output = mlp_layer_norm_output
        return self

    def set_attention_layer_norm_input(self, attention_layer_norm_input):
        self.attention_layer_norm_input = attention_layer_norm_input
        return self

    def set_attention_layer_norm_output(self, attention_layer_norm_output):
        self.attention_layer_norm_output = attention_layer_norm_output
        return self

    def set_norm_after_add(self, norm_after_add):
        self.norm_after_add = norm_after_add
        return self

    def build(self):
        return Transformer(
            d_model=self.d_model,
            num_heads=self.num_heads,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            pos_encoding=self.pos_encoding,
            transformer_layers=self.transformer_layers,
            transformer_norm_type=self.transformer_norm_type,
            transformer_split_self_cross=self.transformer_split_self_cross,
            transformer_parallel_self_cross=self.transformer_parallel_self_cross,
            transformer_message_pass=self.transformer_message_pass,
            transformer_mlp_input=self.transformer_mlp_input,
            transformer_final_projection=self.transformer_final_projection,
            transformer_mlp_bias=self.transformer_mlp_bias,
            transformer_mlp_activation=self.transformer_mlp_activation,
            mlp_layer_norm_input=self.mlp_layer_norm_input,
            mlp_layer_norm_output=self.mlp_layer_norm_output,
            transformer_attention_bias=self.transformer_attention_bias,
            attention_layer_norm_input=self.attention_layer_norm_input,  # prenorm
            attention_layer_norm_output=self.attention_layer_norm_output,
            norm_after_add=self.norm_after_add,
        )