"""
nn.py — DeepSurge (LR wind + stations only).

Derived from the earlier nn3 architecture with HR wind (CONUS404) and HYCOM ocean dropped per the
current project scope. Only two modalities remain:
  - low_res: ERA5 LR wind + surface pressure (U10, V10, PSFC), (B, T, 3, 64, 64)
  - stations: tide-gauge inputs (tide_pred, obs_wl, tide_avail, obs_avail, lat, lon),
              (B, T, num_stations, 6)

The MultimodalFusion layer now carries num_modalities=2 and num_fuse_tokens=2.
missing_mask is dropped from the public input contract; per-station availability
flags inside the station tensor still drive dynamic masking in StationEncoder.

Output: (B, forecast_hours, num_stations, 2) = (mean, variance).
"""

import os
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================
# 1. SPATIAL ENCODER (CNN + CBAM + ViT tokens)
# ==========================================
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        reduced = max(in_channels // reduction_ratio, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, reduced, bias=False),
            nn.ReLU(),
            nn.Linear(reduced, in_channels, bias=False),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.mlp(self.avg_pool(x).view(b, c))
        max_out = self.mlp(self.max_pool(x).view(b, c))
        attn = torch.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * attn


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * attn


class CBAMResBlock(nn.Module):
    """
    Double Conv + CBAM + Residual Skip Connection.
    CBAM applied after ReLU for correct attention computation on positive features.
    GroupNorm (not BatchNorm) is used so that running statistics are not corrupted
    by zero-padded frames in the flattened (B*T) batch.
    """
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.gn1 = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()
        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(num_groups=8, num_channels=out_channels),
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        identity = self.residual(x)
        out = F.relu(self.gn1(self.conv1(x)))
        out = F.relu(self.gn2(self.conv2(out)))
        out = self.ca(out)
        out = self.sa(out)
        out = self.dropout(out)
        return F.relu(out + identity)


class SpatialEncoder(nn.Module):
    """
    Spatial Encoder (CNN + CBAM + ViT Tokenizer) for Gridded Data.

    WHAT IT DOES:
    This encoder looks at a 2D atmospheric map (ERA5 LR wind + surface pressure)
    at a single snapshot in time. First, it uses convolutional layers with attention
    (CBAM) to hunt for important storm physics — like the eye of a hurricane, tight
    wind gradients, or pressure gradients. Then it takes that high-level feature map
    and chops it into a fixed grid (like a chessboard). It unrolls this chessboard
    into a sequence of "visual tokens" and gives each token a permanent 2D coordinate
    ID (positional embedding) so the model knows top-left from bottom-right. It turns
    a weather map into a "sentence" of weather tokens!

    DIMENSION-WISE SUMMARY:
    Let N = B * T (flattened batch and time), C = Input Channels (3: U10, V10, PSFC),
    H = Height, W = Width, d = Hidden dimension (e.g., 256),
    g = Token Grid size (e.g., 6).

    1. Input:
       - IN:  (N, C, H, W) -> (N, 3, 64, 64) for ERA5 LR wind + pressure.
       - ACT: The raw spatiotemporal grids are fed into the network.

    2. Convolutional Extraction & Attention (CBAMResBlocks):
       - IN:  (N, C, H, W)
       - ACT: Passes through 4 residual blocks. Strides physically shrink the map
              (downsampling), while the channel count expands to 'd'. CBAM attention
              helps the network focus on severe weather anomalies and ignore calm
              areas. Default strides (2, 2, 1, 1) reduce 64x64 to 16x16 before pooling.
       - OUT: (N, d, H', W') -> The spatial dimensions are now much smaller.

    3. Spatial Pooling (The "Chessboard" Chop):
       - IN:  (N, d, H', W')
       - ACT: Adaptive pooling aggressively forces whatever spatial resolution is left
              into a strict, uniform g x g grid (e.g., 6x6 = 36 squares).
       - OUT: (N, d, g, g) -> e.g., (N, 128, 6, 6).

    4. Tokenization (Flattening):
       - IN:  (N, d, g, g)
       - ACT: Unrolls the 2D grid into a 1D sequence of tokens. It swaps the axes so
              the 'word' length is the sequence, and 'd' is the feature size.
       - OUT: (N, g*g, d) -> e.g., (N, 36, 256).

    5. Positional Embedding & Normalization:
       - IN:  (N, g*g, d) + (1, g*g, d)
       - ACT: Because we unrolled the 2D map, it forgot its geometry! We add a
              learned, static positional embedding to inject the X/Y map coordinates
              back into the tokens. Finally, we apply LayerNorm for Transformer
              stability.
       - OUT: (N, g*g, d) -> The final sequence of weather tokens, ready for fusion!
    """
    def __init__(self, in_channels=3, d_model=256, token_grid=6, dropout=0.1, strides=(2, 2, 1, 1)):
        super().__init__()
        self.token_grid = token_grid
        self.expected_channels = in_channels

        channels = [16, 32, 64, 128]
        self.layer1 = CBAMResBlock(in_channels, channels[0], stride=strides[0], dropout=dropout)
        self.layer2 = CBAMResBlock(channels[0], channels[1], stride=strides[1], dropout=dropout)
        self.layer3 = CBAMResBlock(channels[1], channels[2], stride=strides[2], dropout=dropout)
        self.layer4 = CBAMResBlock(channels[2], channels[3], stride=strides[3], dropout=dropout)

        self.spatial_pool = nn.AdaptiveAvgPool2d(token_grid)
        self.feature_proj = nn.Linear(channels[3], d_model)
        self.token_norm = nn.LayerNorm(d_model)

        self.pos_embed = nn.Parameter(torch.randn(1, token_grid * token_grid, d_model) * 0.02)

    def forward(self, x):
        if x.shape[1] != self.expected_channels:
            raise ValueError(
                f"SpatialEncoder expects {self.expected_channels} channels at dim=1, "
                f"got shape {x.shape}."
            )
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.spatial_pool(x)
        N, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(N, H * W, C)
        x = self.feature_proj(x)
        x = x + self.pos_embed
        x = self.token_norm(x)
        return x


# ==========================================
# 2. STATION ENCODER
# ==========================================
class StationEncoder(nn.Module):
    """
    Graph Neural Network (Transformer) for Tide Gauge Stations.

    WHAT IT DOES:
    This encoder processes tide gauge stations at every snapshot in time. It treats
    the stations as a dynamic graph: stations appear and disappear based on sensor
    health. Crucially, it performs **dynamic, timestep-wise masking**: for every
    single timestamp it identifies which stations are broken and treats them as
    "non-existent" so the attention mechanism never leaks information from a broken
    sensor to a working one. It gives each station a permanent "ID badge"
    (embedding) to encode local bathymetry, then uses self-attention to route surge
    warnings between stations.

    DIMENSION-WISE SUMMARY:
    Let B = Batch size, T = Time steps, S = Number of stations, d = Hidden dim (e.g., 256).
    Note: The input is already time-flattened to (B*T) before entering this module.

    1. Input & Gating:
       - IN:  (B*T, S, 6) -> [tide_pred, obs_wl, tide_avail, obs_avail, lat, lon]
       - ACT: Multiplies signal by availability flags. Uses the boolean `station_mask`
              (True = sensor broken) to zero out dead rows entirely.
       - OUT: (B*T, S, 6)

    2. Linear Projection:
       - IN:  (B*T, S, 6)
       - ACT: An MLP inflates the raw features into a rich hidden space.
       - OUT: (B*T, S, d)

    3. Geographic Identity (Static Embeddings):
       - IN:  (B*T, S, d) + (1, S, d)
       - ACT: Injects static bathymetry/location embeddings into the live signal.
       - OUT: (B*T, S, d)

    4. Spatial Self-Attention (The Graph Math):
       - IN:  (B*T, S, d)
       - ACT: Multi-head attention calculates surge interactions. The `station_mask`
              is passed as a `key_padding_mask`, mathematically blocking broken
              sensors. When a row is *fully* masked we temporarily un-mask the row
              (`safe_mask = station_mask & ~all_masked`) to avoid a -inf softmax that
              would NaN the backward pass, then zero the resulting attention output.
       - OUT: (B*T, S, d)

    5. Feed-Forward & Cleanup:
       - IN:  (B*T, S, d)
       - ACT: Final non-linear refinement. Explicitly zeroes out the token vectors
              of offline stations so no residual noise leaks into fusion.
       - OUT: (B*T, S, d) -> Context-aware station tokens ready for fusion!
    """
    def __init__(self, num_stations, in_features=6, d_model=256, nhead=4, dropout=0.1):
        super().__init__()
        self.num_stations = num_stations

        self.feature_proj = nn.Sequential(
            nn.Linear(in_features, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.station_embeddings = nn.Parameter(torch.randn(1, num_stations, d_model) * 0.02)

        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, station_mask=None):
        """
        Args:
            x: (B*T, num_stations, 6) — station features per timestep
            station_mask: (B*T, num_stations) bool — True = station data missing
        Returns:
            (B*T, num_stations, d_model)
        """
        tide_flag = (x[..., 2:3] >= 0.5).float()
        obs_flag = (x[..., 3:4] >= 0.5).float()

        x = x.clone()
        x[..., 0:1] = x[..., 0:1] * tide_flag
        x[..., 1:2] = x[..., 1:2] * obs_flag

        if station_mask is not None:
            x = x * (~station_mask).unsqueeze(-1).float()

        h = self.feature_proj(x) + self.station_embeddings

        h_norm = self.norm1(h)
        # Rows that are fully masked produce NaNs on the backward pass of
        # MultiheadAttention even if we nan_to_num the forward. Unmask such
        # rows for the attention call, then zero them out afterwards.
        if station_mask is not None:
            all_masked = station_mask.all(dim=-1, keepdim=True)
            safe_mask = station_mask & ~all_masked
        else:
            safe_mask = None

        h_attn, _ = self.attn(h_norm, h_norm, h_norm, key_padding_mask=safe_mask)
        if station_mask is not None:
            h_attn = torch.nan_to_num(h_attn, nan=0.0)
        h = h + h_attn
        h = h + self.ffn(self.norm2(h))
        if station_mask is not None:
            h = h * (~station_mask).unsqueeze(-1).float()
        return h


# ==========================================
# 3. MULTIMODAL FUSION (2 modalities: lr_wind + stations)
# ==========================================
class MultimodalFusion(nn.Module):
    """
    Multi-Query Multimodal Fusion Transformer (The "Global Storm" Summarizer).

    WHAT IT DOES:
    This layer acts as a conference room for a single snapshot in time. It takes the
    "sentence" of weather tokens (from the SpatialEncoder) and the tide gauge tokens
    (from the StationEncoder) and dumps them all into one long sequence. To ensure
    the network knows which token is which, it stamps each token with a "Modality
    Badge" (e.g., "I am ERA5 LR wind", "I am tide-gauge station").

    Instead of a single [FUSE] token that must compress all ~56 tokens (LR + stations)
    into 1 vector, we use K learnable query tokens (Perceiver-style), where K = number
    of active modalities. Each query token specializes as a "representative" for its
    modality's contribution. This widens the information bandwidth entering the
    temporal transformer by a factor of K.

    DIMENSION-WISE SUMMARY:
    Let N = B * T (flattened batch and time), d = Hidden dimension (e.g., 256).
    Let K = num_fuse_tokens (e.g., 2 for low_res + stations).
    Let L1 = LR wind tokens (e.g., 36), L2 = station tokens (e.g., 20 = num_stations).
    Let L_total = L1 + L2 (e.g., 56).

    1. Modality Tagging (Embeddings):
       - IN:  List of 2 tensors, shapes (N, L_i, d)
       - ACT: Adds a unique, learnable 1D vector to all tokens of a specific modality
              so the Transformer doesn't confuse a wind token with a station token.
       - OUT: List of 2 tensors, shapes (N, L_i, d)

    2. Sequence Assembly (Concatenation):
       - IN:  K [FUSE] query tokens (N, K, d) + the 2 tagged modality tensors.
       - ACT: Glues them all together horizontally into one 1D sequence. It also
              glues the missing-data masks together, ensuring the K [FUSE] tokens
              are never masked (False/0).
       - OUT: (N, K + L_total, d) -> e.g., (N, 58, 256) for K=2

    3. Cross-Modal Self-Attention (The Transformer):
       - IN:  (N, K + L_total, d) plus the combined boolean mask.
       - ACT: Multi-head attention lets atmospheric pressure gradients mathematically
              link with local tide gauges. The K query tokens attend to all modality
              tokens AND to each other. Tokens flagged as missing (True in the mask —
              e.g., offline stations) are completely ignored.
       - OUT: (N, K + L_total, d) -> The fully context-aware sequence.

    4. Multi-Query Extraction:
       - IN:  (N, K + L_total, d)
       - ACT: Throw away the ~56 weather and station tokens. Slice out the first K
              tokens (the [FUSE] query tokens), each of which has absorbed a
              different aspect of the cross-modal information.
       - OUT: (N, K, d) -> e.g., (N, 2, 256). K "Storm State" vectors per timestep!
    """
    def __init__(self, d_model=256, num_modalities=2, num_fuse_tokens=2, num_layers=2, nhead=8, dropout=0.1):
        super().__init__()
        self.num_fuse_tokens = num_fuse_tokens
        self.modality_embeddings = nn.ParameterList(
            [nn.Parameter(torch.randn(1, 1, d_model) * 0.02) for _ in range(num_modalities)]
        )
        self.fusion_tokens = nn.Parameter(torch.randn(1, num_fuse_tokens, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True,
            dim_feedforward=d_model * 4, dropout=dropout, norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, modality_tokens, modality_masks):
        """
        Args:
            modality_tokens: list of 2 tensors [(B*T, L_lr, d), (B*T, L_st, d)]
            modality_masks:  list of 2 bool tensors [(B*T, L_lr), (B*T, L_st)] — True = masked
        Returns:
            (B*T, K, d_model) — the K [FUSE] query token outputs
        """
        b_t = modality_tokens[0].size(0)
        K = self.num_fuse_tokens

        tagged = [tokens + self.modality_embeddings[i] for i, tokens in enumerate(modality_tokens)]
        fuse = self.fusion_tokens.expand(b_t, -1, -1)
        sequence = torch.cat([fuse] + tagged, dim=1)

        fuse_mask = torch.zeros((b_t, K), dtype=torch.bool, device=sequence.device)
        full_mask = torch.cat([fuse_mask] + modality_masks, dim=1)

        attended = self.transformer_encoder(sequence, src_key_padding_mask=full_mask)
        return attended[:, :K, :]


# ==========================================
# 4. CROSS-ATTENTION STATION HEAD
# ==========================================
class StationAwareHead(nn.Module):
    """
    Cross-Attention Station-Aware Output Head (The Localizer).

    WHAT IT DOES:
    This head acts as the final translator. It receives the K temporal tokens per
    forecast hour from the Temporal Transformer. Instead of broadcasting an identical
    global vector to all stations, each station's permanent "ID Badge" (static
    embedding) acts as a QUERY that cross-attends into the K temporal tokens. This
    lets each station selectively extract the storm aspects most relevant to its
    local physics — e.g., a Gulf Coast station might focus on the wind-dominated
    fuse token, while a station near an inlet might focus on the station-dominated
    token.

    DIMENSION-WISE SUMMARY:
    Let B = Batch size, T = Forecast sequence length (e.g., 48),
    K = Number of fuse tokens per timestep (e.g., 2),
    S = Number of stations (e.g., 20), H = Hidden dimension (e.g., 256).

    1. Reshape Temporal Tokens:
       - IN:  (B, T * K, H) -> The interleaved output from the Temporal Transformer.
       - ACT: Reshape to (B*T, K, H) so we process each forecast hour independently.
       - OUT: (B*T, K, H) -> keys/values for cross-attention.

    2. Station Queries:
       - IN:  (1, S, H) -> The permanent geographic identities.
       - ACT: Expand to (B*T, S, H) so each station queries each forecast hour.
       - OUT: (B*T, S, H) -> queries for cross-attention.

    3. Cross-Attention (The Selective Extraction):
       - IN:  Q=(B*T, S, H), K=V=(B*T, K, H)
       - ACT: Each station's embedding attends to the K fuse tokens. Attention
              weights are learned, so each station can specialize in which fuse
              token(s) it pulls information from. Pre-norm for stability.
       - OUT: (B*T, S, H) -> station-specific context per forecast hour.

    4. The Probabilistic Projection:
       - IN:  (B*T, S, H)
       - ACT: A 2-layer MLP projects the cross-attended features down to 2 numbers.
       - OUT: (B, T, S, 2) -> The final predicted (mean, variance) for every station
              at every hour!
    """
    def __init__(self, hidden_dim, num_stations, num_fuse_tokens=2, nhead=4, dropout=0.1):
        super().__init__()
        self.num_stations = num_stations
        self.num_fuse_tokens = num_fuse_tokens

        self.station_embedding = nn.Parameter(torch.randn(1, num_stations, hidden_dim) * 0.02)
        self.norm_q = nn.LayerNorm(hidden_dim)
        self.norm_kv = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=nhead, batch_first=True, dropout=dropout,
        )
        self.norm_ff = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # Split projection: proj_mean is part of the "mean path" and is frozen
        # in phase 2; proj_var is the only trainable head during variance
        # training. See train.py phase-2 freeze logic.
        self.proj_mean = nn.Linear(hidden_dim, 1)
        self.proj_var = nn.Linear(hidden_dim, 1)

    def forward(self, decoder_out):
        """
        Args:
            decoder_out: (B, T_forecast * K, hidden_dim) — interleaved temporal tokens
        Returns:
            (B, T_forecast, num_stations, 2)
        """
        B = decoder_out.shape[0]
        K = self.num_fuse_tokens
        T = decoder_out.shape[1] // K
        H = decoder_out.shape[2]

        kv = decoder_out.reshape(B * T, K, H)
        queries = self.station_embedding.expand(B * T, -1, -1)

        q_norm = self.norm_q(queries)
        kv_norm = self.norm_kv(kv)
        attn_out, _ = self.cross_attn(q_norm, kv_norm, kv_norm)
        h = queries + attn_out
        h = h + self.ffn(self.norm_ff(h))

        mean_out = self.proj_mean(h)
        var_out = self.proj_var(h)
        out = torch.cat([mean_out, var_out], dim=-1)
        return out.reshape(B, T, self.num_stations, 2)


class StationMLPHead(nn.Module):
    """
    Shared-MLP Output Head (Defensibility Baseline for the Cross-Attention Head).

    WHAT IT DOES:
    This head is the "no-novelty" alternative to the cross-attention output head
    (T3e ablation). For each forecast hour it takes the K fused temporal tokens,
    flattens them into one wide vector, runs them through a small shared MLP, and
    projects directly to a (num_stations × 2) prediction grid. There is no
    per-station learned query embedding and no attention — each station's
    prediction is just a column of the final linear layer. Station identity is
    implicit in the output dimension, not in a learned identity token. If this
    baseline matches or beats the cross-attention head on MAE, it tells us that
    per-station selective extraction wasn't paying for itself at this scale.

    DIMENSION-WISE SUMMARY:
    Let B = Batch size, T = Forecast sequence length (e.g., 48),
    K = Number of fuse tokens per timestep (e.g., 2),
    S = Number of stations (e.g., 20), H = Hidden dimension (e.g., 128).

    1. Reshape Temporal Tokens (Flatten the K Fuse Slots):
       - IN:  (B, T * K, H) -> The interleaved output from the Temporal Transformer.
       - ACT: Reshape to (B, T, K * H). The K tokens at each forecast hour become
              one wide feature vector per hour — no attention will sort them out,
              they go in flattened.
       - OUT: (B, T, K * H) -> e.g., (B, 48, 256) for K=2, H=128.

    2. Pre-norm + Shared MLP Body:
       - IN:  (B, T, K * H)
       - ACT: LayerNorm over the K*H feature axis, then a single Linear -> GELU
              -> Dropout block. Same MLP weights apply at every forecast hour and
              every batch element.
       - OUT: (B, T, K * H) -> shared per-hour feature vector.

    3. Per-Station Linear Projection (Mean + Variance):
       - IN:  (B, T, K * H)
       - ACT: Two parallel Linear layers project the (K*H)-dim feature vector to
              S outputs. proj_mean's column for station s plays the role that the
              learned station embedding played in the cross-attn head — it is the
              only place where station identity lives. proj_mean / proj_var stay
              separate so the phase-2 freeze contract (mean frozen, variance
              trained) still works the same way.
       - OUT: Two tensors of shape (B, T, S).

    4. Stack to Gaussian Output:
       - IN:  mean_out, var_out — both (B, T, S)
       - ACT: Stack along a new last axis.
       - OUT: (B, T, S, 2) -> The final predicted (mean, raw_variance) for every
              station at every forecast hour. Variance softplus happens in the
              parent forecaster, identical to the cross-attn head.
    """
    def __init__(self, hidden_dim, num_stations, num_fuse_tokens=2, dropout=0.1):
        super().__init__()
        self.num_stations = num_stations
        self.num_fuse_tokens = num_fuse_tokens
        in_dim = hidden_dim * num_fuse_tokens
        self.norm = nn.LayerNorm(in_dim)
        self.body = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.proj_mean = nn.Linear(in_dim, num_stations)
        self.proj_var = nn.Linear(in_dim, num_stations)

    def forward(self, decoder_out):
        B = decoder_out.shape[0]
        K = self.num_fuse_tokens
        T = decoder_out.shape[1] // K
        H = decoder_out.shape[2]
        flat = decoder_out.reshape(B, T, K * H)
        h = self.body(self.norm(flat))
        mean_out = self.proj_mean(h)
        var_out = self.proj_var(h)
        return torch.stack([mean_out, var_out], dim=-1)


# ==========================================
# 5. MAIN FORECASTER
# ==========================================
class DeepSurgeForecaster(nn.Module):
    """
    Multi-Query Bidirectional Temporal Transformer Forecaster (The Grand Orchestrator).

    WHAT IT DOES:
    This is the master module that pieces the entire physical puzzle together. It
    ingests 72 continuous hours of raw LR wind + tide gauge data. First, it processes
    every single hour independently, using K learnable query tokens (one per active
    modality) to capture K distinct aspects of the coastal state.

    Then, it utilizes a Bidirectional Temporal Transformer to track the storm over
    time. It processes 72*K tokens simultaneously with dual positional embeddings
    (temporal hour + intra-step index). This allows the model to track historical
    momentum (past 24h) while simultaneously scanning the known future weather
    trajectory (future 48h) to anticipate physical storm surges before they happen.

    Finally, a Cross-Attention Output Head lets each station selectively query the
    K temporal tokens per forecast hour, extracting the storm aspects most relevant
    to its local physics.

    KEY DESIGN DECISION — WHY NO CAUSAL MASK:
    Unlike language models where future tokens are unknown, our future tokens contain
    legitimate weather FORECASTS (not ground-truth targets). The only "answer" that
    could cause data leakage — future observed water levels — is explicitly wiped to
    zero before entering the model (see forward()). This physical data wipeout makes
    the causal mask both redundant and harmful:
    - Redundant: No target information exists in future tokens.
    - Harmful: It prevents the model from seeing incoming storms, which is the
      single most important signal for surge prediction.

    DIMENSION-WISE SUMMARY:
    Let B = Batch size, T_lookback = 24, T_forecast = 48, T_total = 72.
    K = Number of fuse tokens per timestep (e.g., 2).
    S = Number of Stations, d = Hidden dimension (e.g., 256).

    1. The Inputs (Raw Data):
       - IN:  Low-Res wind (B, 72, 3, 64, 64), Stations (B, 72, S, 6).
       - ACT: Derives a per-hour station mask from the availability flags and
              explicitly wipes out future station observations so the model
              cannot cheat during training.

    2. Spatial Encoding & Multi-Query Fusion:
       - IN:  The raw 72-hour inputs.
       - ACT:
           a) Physical Wipeout: Future obs_wl / obs_avail are zeroed past the
              lookback boundary, so no target information can leak.
           b) Mathematical Wipeout: The Fusion Transformer uses a boolean station
              mask to set attention scores for offline tide gauges to -inf.
           c) K query tokens absorb cross-modal information per timestep.
       - OUT: (B, 72*K, d) -> fused_sequence (K context-aware tokens per hour).

    3. Dual Positional Embedding:
       - IN:  (B, 72*K, d)
       - ACT: Adds two learnable embeddings: temporal position (which hour?) and
              intra-step position (which of the K tokens?). Analogous to 2D ViT
              positional encoding.
       - OUT: (B, 72*K, d) -> position-aware fused sequence.

    4. Bidirectional Temporal Transformer (Full Timeline Processing):
       - IN:  (B, 72*K, d)
       - ACT: Processes all 72*K tokens bidirectionally. Tokens within the same
              hour attend to each other (intra-step mixing), tokens across hours
              attend to each other (temporal reasoning), and all combinations
              between past/future and across modality representatives happen
              simultaneously.
       - OUT: (B, 72*K, d) -> temporal_out (fully contextualized timeline).

    5. Forecast Slicing:
       - IN:  (B, 72*K, d)
       - ACT: Slice out only the 48*K forecast tokens.
       - OUT: (B, 48*K, d) -> forecast_seq_out.

    6. Cross-Attention Output Head:
       - IN:  (B, 48*K, d)
       - ACT: Each station's static embedding queries the K tokens per forecast
              hour via cross-attention, then projects to a Gaussian distribution.
       - OUT: (B, 48, S, 2) -> The final probabilistic prediction!
    """
    def __init__(self, num_stations, lookback_hours=24, forecast_hours=48,
                 d_model=256, hidden_dim=512, token_grid=6, num_fuse_tokens=2, dropout=0.1,
                 head_type='cross_attn'):
        super().__init__()
        self.num_stations = num_stations
        self.token_grid = token_grid
        self.num_fuse_tokens = num_fuse_tokens
        self.lookback_hours = lookback_hours
        self.forecast_hours = forecast_hours
        self.head_type = head_type

        self.enc_low_res = SpatialEncoder(
            in_channels=3, d_model=d_model, token_grid=token_grid,
            dropout=dropout, strides=(2, 2, 1, 1),
        )
        self.enc_stations = StationEncoder(
            num_stations=num_stations, in_features=6, d_model=d_model, dropout=dropout,
        )

        self.fusion = MultimodalFusion(
            d_model=d_model, num_modalities=2, num_fuse_tokens=num_fuse_tokens,
            num_layers=2, dropout=dropout,
        )

        T_total = lookback_hours + forecast_hours
        self.time_pos_embed = nn.Parameter(torch.randn(1, T_total, d_model) * 0.02)
        self.fuse_pos_embed = nn.Parameter(torch.randn(1, num_fuse_tokens, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=8, dim_feedforward=hidden_dim,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        if head_type == 'cross_attn':
            self.output_head = StationAwareHead(
                hidden_dim=d_model, num_stations=num_stations,
                num_fuse_tokens=num_fuse_tokens, dropout=dropout,
            )
        elif head_type == 'mlp':
            self.output_head = StationMLPHead(
                hidden_dim=d_model, num_stations=num_stations,
                num_fuse_tokens=num_fuse_tokens, dropout=dropout,
            )
        else:
            raise ValueError(f"Unknown head_type={head_type!r}; expected 'cross_attn' or 'mlp'.")

    def _time_distribute_encode(self, x_lr, x_st, station_mask):
        """
        Time-Distributed Spatial Encoding.

        WHAT IT DOES:
        Our spatial CNN and graph encoders are designed to look at a single snapshot
        in time (like a single photograph), but our input data is a 72-hour continuous
        video. This function acts as a "film unroller." It collapses the Batch and
        Time dimensions together, tricking the spatial encoders into treating the
        72-hour sequences as one massive pile of independent, disconnected images.

        HOW THE LOGIC FLOWS:

        1. Flattening the Timeline (Unrolling):
           We merge the Batch and Time dimensions. If we have a batch of 8 samples,
           each with 72 hours, we reshape the data to look like 576 separate snapshots.

        2. Spatial Encoding (LR wind):
           The unrolled LR wind grids are pushed through the SpatialEncoder in one
           shot, producing (B*T, 36, d) visual tokens. LR wind is always present in
           the current pipeline, so there is no sparse/masked dispatch — the encoder
           runs on every frame.

        3. Station Encoding:
           Station data is never fully missing for all locations at once. We pass
           the unrolled station data through the StationEncoder and rely on
           `station_mask` to zero out the final tokens of any broken/missing sensors.

        4. Multimodal Fusion:
           We bundle the unrolled LR wind and station tokens (plus their masks) and
           hand them to the Fusion layer. The Fusion layer lets the two modalities
           cross-reference each other to summarize the storm's physics for each
           independent hour, producing K fuse tokens per timestep.

        5. Re-rolling the Film:
           Finally, we separate the 576 snapshots back out into 8 samples of 72 hours
           and interleave the K fuse tokens into a (B, T*K, d) sequence ready for the
           Temporal Transformer.
        """
        B, T = x_lr.shape[0], x_lr.shape[1]
        B_T = B * T
        K = self.num_fuse_tokens

        expected_T = self.lookback_hours + self.forecast_hours
        if T != expected_T:
            raise ValueError(
                f"Input sequence length T={T} does not match "
                f"lookback ({self.lookback_hours}) + forecast ({self.forecast_hours})"
            )

        x_lr = x_lr.reshape(B_T, *x_lr.shape[2:])
        x_st = x_st.reshape(B_T, *x_st.shape[2:])

        st_mask = station_mask.reshape(B_T, -1) if station_mask is not None else None

        num_tokens = self.token_grid * self.token_grid
        d_model = self.enc_low_res.feature_proj.out_features

        v_lr = self.enc_low_res(x_lr)                                  # (B_T, L_lr, d)
        v_st = self.enc_stations(x_st, st_mask)                        # (B_T, S, d)

        lr_mask = torch.zeros((B_T, num_tokens), dtype=torch.bool, device=v_lr.device)
        if st_mask is None:
            st_mask_full = torch.zeros((B_T, self.num_stations), dtype=torch.bool, device=v_lr.device)
        else:
            st_mask_full = st_mask

        fused = self.fusion([v_lr, v_st], [lr_mask, st_mask_full])     # (B_T, K, d)
        fused = fused.reshape(B, T, K, d_model).reshape(B, T * K, d_model)
        return fused

    def forward(self, inputs):
        """
        Args:
            inputs: dict with
                'low_res'  : (B, T_total, 3, 64, 64) — ERA5 LR wind + PSFC
                'stations' : (B, T_total, num_stations, 6)
                             channels: (tide_pred, obs_wl, tide_avail, obs_avail, lat, lon)
        Returns:
            (B, forecast_hours, num_stations, 2) — (mean, variance) per station per hour.
        """
        stations = inputs['stations'].clone()

        tide_avail = stations[..., 2]
        obs_avail = stations[..., 3]
        station_mask = (obs_avail < 0.5) & (tide_avail < 0.5)

        # Future observation wipeout (no causal mask needed).
        stations[:, self.lookback_hours:, :, 1] = 0.0
        stations[:, self.lookback_hours:, :, 3] = 0.0
        station_mask[:, self.lookback_hours:, :] = stations[:, self.lookback_hours:, :, 2] < 0.5

        fused_sequence = self._time_distribute_encode(inputs['low_res'], stations, station_mask)

        K = self.num_fuse_tokens
        T_total = self.lookback_hours + self.forecast_hours
        time_pos = self.time_pos_embed.repeat_interleave(K, dim=1)
        fuse_pos = self.fuse_pos_embed.repeat(1, T_total, 1)
        fused_sequence = fused_sequence + time_pos + fuse_pos

        temporal_out = self.temporal_transformer(fused_sequence)
        forecast_seq_out = temporal_out[:, self.lookback_hours * K:, :]

        out = self.output_head(forecast_seq_out)
        mean = out[..., 0]
        variance = F.softplus(out[..., 1]) + 1e-6
        return torch.stack([mean, variance], dim=-1)


# ==========================================
# 6. LOSSES
# ==========================================
class SurgeDeterministicLoss(nn.Module):
    """
    Phase 1 Loss Function: Evaluates only the deterministic mean, but KEEPS the
    physical safety weights (event multiplier + asymmetric under-prediction
    penalty) so the backbone learns to track storms. Use this to pre-train the
    mean before switching to the probabilistic NLL head.
    """
    def __init__(self, scaler_path, surge_threshold_meters=0.5, event_multiplier=2.0,
                 asymmetry_factor=1.5, use_weighted_loss=True):
        super().__init__()
        self.surge_threshold_meters = surge_threshold_meters
        self.event_multiplier = event_multiplier
        self.asymmetry_factor = asymmetry_factor
        self.use_weighted_loss = use_weighted_loss

        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler file not found at {scaler_path}")
        with open(scaler_path, 'rb') as f:
            params = pickle.load(f)
        if params['method'] != 'zscore':
            raise ValueError("SurgeDeterministicLoss expects a 'zscore' scaler.")

        self.register_buffer('data_mean', torch.tensor(params['mean'], dtype=torch.float32))
        self.register_buffer('data_std', torch.tensor(params['std'], dtype=torch.float32))

    def forward(self, predictions, targets, target_mask=None):
        mean = predictions[..., 0].float()
        targets = targets.float()
        squared_error = (targets - mean) ** 2

        if self.use_weighted_loss:
            physical_targets = (targets * self.data_std) + self.data_mean
            physical_mean = (mean * self.data_std) + self.data_mean

            excess_surge = torch.relu(physical_targets.abs() - self.surge_threshold_meters)
            event_weights = 1.0 + (self.event_multiplier * excess_surge)
            under_prediction = (physical_targets > physical_mean).float()
            asymmetric_weights = 1.0 + ((self.asymmetry_factor - 1.0) * under_prediction)
            loss_per_element = event_weights * asymmetric_weights * squared_error
        else:
            loss_per_element = squared_error

        if target_mask is not None:
            mask_expanded = target_mask.unsqueeze(1).float()
            masked_loss = loss_per_element * mask_expanded
            T = targets.size(1)
            valid_elements = target_mask.sum() * T
            valid_elements = valid_elements.clamp(min=1.0)
            return masked_loss.sum() / valid_elements
        return loss_per_element.mean()


class SurgeProbabilisticLoss(nn.Module):
    """
    Gaussian NLL with target-only event weighting.

    Weighting choice (deliberate). Event weights multiply the *full* NLL
    (both 0.5*log(var) and 0.5*sq_err/var). For a fixed mean error e and
    variance v, the per-sample optimum of w*(0.5*log(v) + 0.5*e^2/v) is
    v* = e^2 — independent of w — so weighting preserves the calibrated
    optimum and only rescales gradient magnitude. The alternative form
    (weighting only the squared-error term) yields v* = w*e^2, i.e. storm
    samples would converge to inflated variance and bias calibration.
    Do not switch to that form.

    Asymmetric weighting (off by default). The under-prediction multiplier
    depends on physical_targets > physical_mean, which makes the loss a
    prediction-dependent (improper) scoring rule and breaks calibration
    guarantees. Gated behind `use_asymmetric_weighting=False`. Enable only
    as an explicit ablation, never for the calibrated-uncertainty headline
    runs.
    """
    def __init__(self, scaler_path, surge_threshold_meters=0.5, event_multiplier=2.0,
                 asymmetry_factor=1.5, eps=1e-6, use_weighted_loss=True,
                 use_asymmetric_weighting=False):
        super().__init__()
        self.surge_threshold_meters = surge_threshold_meters
        self.event_multiplier = event_multiplier
        self.asymmetry_factor = asymmetry_factor
        self.eps = eps
        self.use_weighted_loss = use_weighted_loss
        self.use_asymmetric_weighting = use_asymmetric_weighting

        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler file not found at {scaler_path}")
        with open(scaler_path, 'rb') as f:
            params = pickle.load(f)
        if params['method'] != 'zscore':
            raise ValueError("SurgeProbabilisticLoss expects a 'zscore' scaler.")

        self.register_buffer('data_mean', torch.tensor(params['mean'], dtype=torch.float32))
        self.register_buffer('data_std', torch.tensor(params['std'], dtype=torch.float32))

    def forward(self, predictions, targets, target_mask=None):
        mean = predictions[..., 0].float()
        variance = predictions[..., 1].float() + self.eps
        targets = targets.float()
        squared_error = (targets - mean) ** 2

        nll = 0.5 * torch.log(variance) + 0.5 * (squared_error / variance)

        if self.use_weighted_loss:
            physical_targets = (targets * self.data_std) + self.data_mean
            excess_surge = torch.relu(physical_targets.abs() - self.surge_threshold_meters)
            weights = 1.0 + (self.event_multiplier * excess_surge)
            if self.use_asymmetric_weighting:
                physical_mean = (mean * self.data_std) + self.data_mean
                under_prediction = (physical_targets > physical_mean).float()
                weights = weights * (1.0 + ((self.asymmetry_factor - 1.0) * under_prediction))
            nll = weights * nll

        if target_mask is not None:
            mask_expanded = target_mask.unsqueeze(1).float()
            nll = nll * mask_expanded
            T = targets.size(1)
            valid_elements = target_mask.sum() * T
            valid_elements = valid_elements.clamp(min=1.0)
            return nll.sum() / valid_elements
        return nll.mean()


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Match the locked-default training config (d_model=128, hidden_dim=128,
    # K=1, lookback=24, forecast=72). Change HEAD_TYPE here to flip the head
    # under test; both are profiled side-by-side below regardless.
    HEAD_TYPE = 'cross_attn'   # 'cross_attn' or 'mlp'
    num_stations = 20
    lookback, forecast = 24, 72
    T_total = lookback + forecast
    d_model, hidden_dim, token_grid, K = 128, 128, 6, 2
    B = 2

    def count_params(module):
        return sum(p.numel() for p in module.parameters() if p.requires_grad)

    def build(head_type, n_st):
        return DeepSurgeForecaster(
            num_stations=n_st, lookback_hours=lookback, forecast_hours=forecast,
            d_model=d_model, hidden_dim=hidden_dim, token_grid=token_grid,
            num_fuse_tokens=K, head_type=head_type,
        )

    # ------------------------------------------------------------------
    # 1. Per-component breakdown for both head variants (num_stations = 20)
    # ------------------------------------------------------------------
    components = ['enc_low_res', 'enc_stations', 'fusion',
                  'temporal_transformer', 'output_head']
    extras = ['time_pos_embed', 'fuse_pos_embed']

    bar = '=' * 78
    print(bar)
    print(f"DeepSurgeForecaster - per-component params - "
          f"d_model={d_model}, hidden_dim={hidden_dim}, K={K}, "
          f"S={num_stations}, T={T_total}")
    print(bar)
    print(f"{'component':<28}{'cross_attn':>16}{'mlp':>16}{'delta (mlp - xa)':>18}")
    print('-' * 78)

    m_xa  = build('cross_attn', num_stations)
    m_mlp = build('mlp',        num_stations)
    for name in components:
        a = count_params(getattr(m_xa, name))
        b = count_params(getattr(m_mlp, name))
        print(f"{name:<28}{a:>16,}{b:>16,}{b - a:>+18,}")
    for name in extras:
        a = getattr(m_xa,  name).numel()
        b = getattr(m_mlp, name).numel()
        print(f"{name:<28}{a:>16,}{b:>16,}{b - a:>+18,}")
    tot_xa, tot_mlp = count_params(m_xa), count_params(m_mlp)
    print('-' * 78)
    print(f"{'TOTAL':<28}{tot_xa:>16,}{tot_mlp:>16,}{tot_mlp - tot_xa:>+18,}")
    print(f"{'mlp / cross_attn':<28}{'':>16}{tot_mlp / tot_xa:>16.3f}")
    print(bar)

    # ------------------------------------------------------------------
    # 2. Head-only sweep across station counts
    # (cross_attn has a per-station learned query → grows linearly in S;
    #  mlp has the mean/var Linear out_features = S → also linear in S.
    #  Compare the constants.)
    # ------------------------------------------------------------------
    print(f"\nOutput head only - params vs num_stations")
    print(f"{'num_stations':<14}{'cross_attn':>16}{'mlp':>16}{'mlp / xa':>14}")
    print('-' * 60)
    for s in (9, 20, 34):
        ah = build('cross_attn', s).output_head
        bh = build('mlp',        s).output_head
        a, b = count_params(ah), count_params(bh)
        print(f"{s:<14}{a:>16,}{b:>16,}{b / a:>14.3f}")

    # ------------------------------------------------------------------
    # 3. Smoke test (forward + backward) on the HEAD_TYPE under test
    # ------------------------------------------------------------------
    print(f"\nSmoke test - head_type='{HEAD_TYPE}'")
    model = build(HEAD_TYPE, num_stations)
    dummy_inputs = {
        'low_res':  torch.randn(B, T_total, 3, 64, 64),
        'stations': torch.randn(B, T_total, num_stations, 6),
    }
    targets = torch.randn(B, forecast, num_stations)
    target_mask = torch.ones(B, num_stations)

    preds = model(dummy_inputs)
    print(f"  forward OK - preds {tuple(preds.shape)}")
    assert preds.shape == (B, forecast, num_stations, 2)

    dummy_scaler_path = "temp_dummy_scaler.pkl"
    with open(dummy_scaler_path, 'wb') as f:
        pickle.dump({'method': 'zscore', 'mean': 0.0, 'std': 1.0}, f)
    try:
        crit = SurgeProbabilisticLoss(scaler_path=dummy_scaler_path)
        loss = crit(preds, targets, target_mask)
        loss.backward()
        print(f"  loss = {loss.item():.4f} - backward OK")
    finally:
        os.remove(dummy_scaler_path)
