import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange
from ...subNets import BertTextEncoder
from ...subNets.transformers_encoder.transformer import TransformerEncoder


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------

class SharedDomainAdapter(nn.Module):
    """Shared Domain Adapter (SDA).

    Projects each modality into a low-dimensional bottleneck Transformer,
    then projects back with a residual connection to produce noise-filtered
    shared representations.
    """

    def __init__(self, input_dim, bottleneck_dim=128, nhead=4, num_layers=2):
        super().__init__()
        self.down_proj = nn.Sequential(
            nn.Linear(input_dim, bottleneck_dim),
            nn.GELU()
        )
        self.norm = nn.LayerNorm(bottleneck_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=bottleneck_dim,
            nhead=nhead,
            dim_feedforward=bottleneck_dim * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.up_proj = nn.Linear(bottleneck_dim, input_dim)

    def forward(self, x, mask=None):
        """x: (seq_len, batch, dim) -> (seq_len, batch, dim)"""
        x = x.permute(1, 0, 2)       # (batch, seq_len, dim)
        residual = x
        x = self.down_proj(x)
        x = self.norm(x)
        x = self.transformer(x, src_key_padding_mask=mask)
        x = self.norm(x)
        x = self.up_proj(x)
        return (x + residual).permute(1, 0, 2)


class GatingMechanism(nn.Module):
    """Gating sub-module inside FGA.

    Dynamically filters modality-specific noise guided by the shared
    semantic anchor, preventing semantic mismatch.
    """

    def __init__(self, embed_dim):
        super().__init__()
        self.shared_affine   = nn.Linear(embed_dim, embed_dim, bias=False)
        self.specific_affine = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(self, specific_feat, shared_feat):
        """Returns (filtered_feat, gate_weight), both (B, T, D)."""
        gate_weight = torch.sigmoid(
            self.shared_affine(shared_feat) + self.specific_affine(specific_feat)
        )
        filtered = shared_feat * gate_weight + specific_feat * (1 - gate_weight)
        return filtered, gate_weight


class MultiHeadCrossAttention(nn.Module):
    """Multi-Head Cross-Attention (MA) inside FGA.

    The shared semantic anchor acts as Query; modality-specific features
    serve as Key and Value.
    """

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim   = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads  = heads
        self.scale  = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.to_q   = nn.Linear(dim, inner_dim, bias=False)
        self.to_k   = nn.Linear(dim, inner_dim, bias=False)
        self.to_v   = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out else nn.Identity()
        )

    def forward(self, q, k, v):
        q = q.unsqueeze(0)
        b, n, _, h = *q.shape, self.heads
        q = self.to_q(q)
        k = self.to_k(k)
        v = self.to_v(v)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = self.attend(dots)
        out  = einsum('b h i j, b h j d -> b h i d', attn, v)
        out  = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out), attn


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net  = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(self.norm(x))


class FeatureGuidedAlignment(nn.Module):
    """Feature-Guided Alignment (FGA) module.

    Constructs a global unified shared semantic anchor (ch) from all three
    modalities, then applies:
      1. GatingMechanism  - filters sentiment-irrelevant noise.
      2. MultiHeadCrossAttention (MA) - aligns specific features to the anchor.
    """

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.1):
        super().__init__()
        self.reduce_dim = nn.Linear(3 * dim, dim)
        self.gate_l = GatingMechanism(dim)
        self.gate_v = GatingMechanism(dim)
        self.gate_a = GatingMechanism(dim)
        self.ma_l   = MultiHeadCrossAttention(dim, heads, dim_head, dropout)
        self.ma_v   = MultiHeadCrossAttention(dim, heads, dim_head, dropout)
        self.ma_a   = MultiHeadCrossAttention(dim, heads, dim_head, dropout)
        self.norm_l = nn.LayerNorm(dim)
        self.norm_v = nn.LayerNorm(dim)
        self.norm_a = nn.LayerNorm(dim)
        self.ffn    = FeedForward(dim, dim * 4, dropout=dropout)

    def forward(self, shared_feat_l, shared_feat_v, shared_feat_a,
                spec_feat_l, spec_feat_v, spec_feat_a):
        """
        Args:
            shared_feat_*: (B, T, D)  refined shared representations from SDA
            spec_feat_*:   (B, T, D)  modality-specific representations
        Returns:
            cross_l/v/a, g_l/v/a, attn_l/v/a, shared_anchor
        """
        # Build global shared semantic anchor ch
        shared_anchor = self.reduce_dim(
            torch.cat((shared_feat_l, shared_feat_v, shared_feat_a), dim=-1)
        )
        # Gating: filter noise from specific features
        spec_feat_l, g_l = self.gate_l(spec_feat_l, shared_anchor)
        spec_feat_v, g_v = self.gate_v(spec_feat_v, shared_anchor)
        spec_feat_a, g_a = self.gate_a(spec_feat_a, shared_anchor)
        spec_feat_l = self.norm_l(spec_feat_l)
        spec_feat_v = self.norm_v(spec_feat_v)
        spec_feat_a = self.norm_a(spec_feat_a)
        # MA: cross-attention alignment
        cross_l, attn_l = self.ma_l(shared_anchor, spec_feat_l, spec_feat_l)
        cross_v, attn_v = self.ma_v(shared_anchor, spec_feat_v, spec_feat_v)
        cross_a, attn_a = self.ma_a(shared_anchor, spec_feat_a, spec_feat_a)
        cross_l = self.ffn(self.norm_l(cross_l + shared_anchor))
        cross_v = self.ffn(self.norm_v(cross_v + shared_anchor))
        cross_a = self.ffn(self.norm_a(cross_a + shared_anchor))
        return (cross_l, cross_v, cross_a,
                g_l, g_v, g_a,
                attn_l, attn_v, attn_a,
                shared_anchor)


class DynamicPerspectiveFusion(nn.Module):
    """Dynamic Perspective Fusion (DPF) module.

    Implements the rotating dominant-view strategy inspired by human
    cognitive perspective-taking. Each modality alternately serves as the
    primary view; adaptive weights w_m are computed from the shared anchor.
    """

    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.weight_gen_l = nn.Linear(dim, 2)
        self.weight_gen_v = nn.Linear(dim, 2)
        self.weight_gen_a = nn.Linear(dim, 2)
        self.norm_l = nn.LayerNorm(dim)
        self.norm_v = nn.LayerNorm(dim)
        self.norm_a = nn.LayerNorm(dim)
        self.ffn    = FeedForward(dim, dim * 4, dropout=dropout)

    def forward(self, cross_l, cross_v, cross_a, shared_anchor):
        """
        Args:
            cross_l/v/a:   (B, T, D)  FGA-aligned features
            shared_anchor: (B, T, D)  global shared anchor ch
        Returns:
            dyn_l, dyn_v, dyn_a: (B, T, D)
            w_l, w_v, w_a: adaptive weight tensors
        """
        # Language as primary view
        w_l = torch.softmax(self.weight_gen_l(shared_anchor), dim=-1)
        fused_l = (w_l[..., 0:1] * cross_l + w_l[..., 1:2] * cross_a +
                   w_l[..., 0:1] * cross_l + w_l[..., 1:2] * cross_v)
        dyn_l = self.norm_l(self.ffn(self.norm_l(fused_l + shared_anchor)))
        # Video as primary view
        w_v = torch.softmax(self.weight_gen_v(shared_anchor), dim=-1)
        fused_v = (w_v[..., 0:1] * cross_v + w_v[..., 1:2] * cross_l +
                   w_v[..., 0:1] * cross_v + w_v[..., 1:2] * cross_a)
        dyn_v = self.norm_v(self.ffn(self.norm_v(fused_v + shared_anchor)))
        # Audio as primary view
        w_a = torch.softmax(self.weight_gen_a(shared_anchor), dim=-1)
        fused_a = (w_a[..., 0:1] * cross_a + w_a[..., 1:2] * cross_v +
                   w_a[..., 0:1] * cross_a + w_a[..., 1:2] * cross_l)
        dyn_a = self.norm_a(self.ffn(self.norm_a(fused_a + shared_anchor)))
        return dyn_l, dyn_v, dyn_a, w_l, w_v, w_a


class CPC(nn.Module):
    """Contrastive Predictive Coding loss (L_nce).

    Maintains semantic consistency between shared anchor features and
    dynamic perspective features.
    """

    def __init__(self, x_size, y_size, n_layers=1, activation='Tanh'):
        super().__init__()
        self.x_size     = x_size
        self.y_size     = y_size
        self.layers     = n_layers
        self.activation = getattr(nn, activation)

    def forward(self, x, y):
        x      = torch.mean(x, dim=-2)
        y      = torch.mean(y, dim=-2)
        x_pred = y / y.norm(dim=1, keepdim=True)
        x      = x / x.norm(dim=1, keepdim=True)
        pos    = torch.sum(x * x_pred, dim=-1)
        neg    = torch.logsumexp(torch.matmul(x, x_pred.t()), dim=-1)
        return -(pos - neg).mean()


class AutoEncoder(nn.Module):
    """Lightweight auto-encoder for modality reconstruction."""

    def __init__(self, indim, outdim, dropout_rate=None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(indim, outdim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(outdim, indim),
            nn.GELU()
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Main Model
# ---------------------------------------------------------------------------

class HDFN(nn.Module):
    """Hierarchical Dynamic Fusion Network (HDFN) for Multimodal Sentiment Analysis.

    Architecture:
        Input (Text T, Video V, Audio A)
          [1] Temporal Conv1d projection + modality-specific / shared encoding
          [2] Shared Domain Adapter (SDA)  - bottleneck de-noising
          [3] Feature-Guided Alignment (FGA)
                 Gating Mechanism: suppresses sentiment-irrelevant noise
                 Multi-Head Cross-Attention (MA): aligns to shared anchor ch
          [4] Dynamic Perspective Fusion (DPF)
                 Rotating dominant-view strategy
                 Sample-adaptive weight generation w_m
          [5] Joint prediction
                 L_total = L_MSA + L_dp + L_wc + L_nce
    """

    def __init__(self, args):
        super(HDFN, self).__init__()

        if args.use_bert:
            self.text_model = BertTextEncoder(
                use_finetune=args.use_finetune,
                transformers=args.transformers,
                pretrained=args.pretrained
            )
        self.use_bert = args.use_bert
        dst_feature_dims, nheads = args.dst_feature_dim_nheads

        if args.dataset_name == 'mosi':
            if args.need_data_aligned:
                self.len_l, self.len_v, self.len_a = 50, 50, 50
            else:
                self.len_l, self.len_v, self.len_a = 50, 500, 375
        if args.dataset_name == 'mosei':
            if args.need_data_aligned:
                self.len_l, self.len_v, self.len_a = 50, 50, 50
            else:
                self.len_l, self.len_v, self.len_a = 50, 500, 500

        self.orig_d_l, self.orig_d_a, self.orig_d_v = args.feature_dims
        self.d_l = self.d_a = self.d_v = dst_feature_dims
        self.num_heads      = nheads
        self.layers         = args.nlevels
        self.attn_dropout   = args.attn_dropout
        self.attn_dropout_a = args.attn_dropout_a
        self.attn_dropout_v = args.attn_dropout_v
        self.relu_dropout   = args.relu_dropout
        self.embed_dropout  = args.embed_dropout
        self.res_dropout    = args.res_dropout
        self.output_dropout = args.output_dropout
        self.text_dropout   = args.text_dropout
        self.attn_mask      = args.attn_mask

        # ---- Core HDFN modules ----
        self.sda_l = SharedDomainAdapter(self.d_l)
        self.sda_v = SharedDomainAdapter(self.d_v)
        self.sda_a = SharedDomainAdapter(self.d_a)
        self.fga   = FeatureGuidedAlignment(dim=self.d_l)
        self.dpf   = DynamicPerspectiveFusion(dim=self.d_l)
        self.cpc_loss = CPC(x_size=self.d_l, y_size=self.d_l)

        self.dyn_proj = nn.Sequential(
            nn.Linear(50, 32), nn.ReLU(), nn.Linear(32, 1)
        )
        self.lin        = nn.Linear(2 * self.d_l, self.d_l)
        self.reduce_dim = nn.Linear(3 * self.d_l, self.d_l)
        self.auto       = AutoEncoder(self.d_l, self.d_l * 4,
                                      dropout_rate=self.embed_dropout)

        combined_dim_low  = self.d_a
        combined_dim_high = self.d_a
        combined_dim      = (self.d_l + self.d_a + self.d_v) + self.d_l * 3
        output_dim        = 1

        # [1] Temporal Conv1d projections
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l,
                                kernel_size=args.conv1d_kernel_size_l, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a,
                                kernel_size=args.conv1d_kernel_size_a, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v,
                                kernel_size=args.conv1d_kernel_size_v, padding=0, bias=False)

        # [2] Modality-specific encoders
        self.encoder_s_l = self.get_network(self_type='l', layers=self.layers)
        self.encoder_s_v = self.get_network(self_type='v', layers=self.layers)
        self.encoder_s_a = self.get_network(self_type='a', layers=self.layers)
        # Modality-shared encoder
        self.encoder_c   = self.get_network(self_type='l', layers=self.layers)

        # [3] Decoders for reconstruction
        self.decoder_l = nn.Conv1d(self.d_l * 2, self.d_l, kernel_size=1, padding=0, bias=False)
        self.decoder_v = nn.Conv1d(self.d_v * 2, self.d_v, kernel_size=1, padding=0, bias=False)
        self.decoder_a = nn.Conv1d(self.d_a * 2, self.d_a, kernel_size=1, padding=0, bias=False)

        # Cosine-similarity projectors
        self.proj_cosine_l = nn.Linear(
            combined_dim_low * (self.len_l - args.conv1d_kernel_size_l + 1), combined_dim_low)
        self.proj_cosine_v = nn.Linear(
            combined_dim_low * (self.len_v - args.conv1d_kernel_size_v + 1), combined_dim_low)
        self.proj_cosine_a = nn.Linear(
            combined_dim_low * (self.len_a - args.conv1d_kernel_size_a + 1), combined_dim_low)

        self.align_c_l = nn.Linear(
            combined_dim_low * (self.len_l - args.conv1d_kernel_size_l + 1), combined_dim_low)
        self.align_c_v = nn.Linear(
            combined_dim_low * (self.len_v - args.conv1d_kernel_size_v + 1), combined_dim_low)
        self.align_c_a = nn.Linear(
            combined_dim_low * (self.len_a - args.conv1d_kernel_size_a + 1), combined_dim_low)

        self.self_attentions_c_l = self.get_network(self_type='l')
        self.self_attentions_c_v = self.get_network(self_type='v')
        self.self_attentions_c_a = self.get_network(self_type='a')

        self.proj1_c     = nn.Linear(self.d_l * 3, self.d_l * 3)
        self.proj2_c     = nn.Linear(self.d_l * 3, self.d_l * 3)
        self.out_layer_c = nn.Linear(self.d_l * 3, output_dim)

        # [4] Cross-modal attention (kept for shared encoder pipeline)
        self.trans_l_with_a = self.get_network(self_type='la', layers=self.layers)
        self.trans_l_with_v = self.get_network(self_type='lv', layers=self.layers)
        self.trans_a_with_l = self.get_network(self_type='al')
        self.trans_a_with_v = self.get_network(self_type='av')
        self.trans_v_with_l = self.get_network(self_type='vl')
        self.trans_v_with_a = self.get_network(self_type='va')
        self.trans_l_mem    = self.get_network(self_type='l_mem', layers=self.layers)
        self.trans_a_mem    = self.get_network(self_type='a_mem', layers=3)
        self.trans_v_mem    = self.get_network(self_type='v_mem', layers=3)

        # [5] FC heads for shared features
        self.proj1_l_low    = nn.Linear(combined_dim_low * (self.len_l - args.conv1d_kernel_size_l + 1), combined_dim_low)
        self.proj2_l_low    = nn.Linear(combined_dim_low, combined_dim_low * (self.len_l - args.conv1d_kernel_size_l + 1))
        self.out_layer_l_low = nn.Linear(combined_dim_low * (self.len_l - args.conv1d_kernel_size_l + 1), output_dim)
        self.proj1_v_low    = nn.Linear(combined_dim_low * (self.len_v - args.conv1d_kernel_size_v + 1), combined_dim_low)
        self.proj2_v_low    = nn.Linear(combined_dim_low, combined_dim_low * (self.len_v - args.conv1d_kernel_size_v + 1))
        self.out_layer_v_low = nn.Linear(combined_dim_low * (self.len_v - args.conv1d_kernel_size_v + 1), output_dim)
        self.proj1_a_low    = nn.Linear(combined_dim_low * (self.len_a - args.conv1d_kernel_size_a + 1), combined_dim_low)
        self.proj2_a_low    = nn.Linear(combined_dim_low, combined_dim_low * (self.len_a - args.conv1d_kernel_size_a + 1))
        self.out_layer_a_low = nn.Linear(combined_dim_low * (self.len_a - args.conv1d_kernel_size_a + 1), output_dim)

        # [6] FC heads for specific features
        self.proj1_l_high    = nn.Linear(combined_dim_high, combined_dim_high)
        self.proj2_l_high    = nn.Linear(combined_dim_high, combined_dim_high)
        self.out_layer_l_high = nn.Linear(combined_dim_high, output_dim)
        self.proj1_v_high    = nn.Linear(combined_dim_high, combined_dim_high)
        self.proj2_v_high    = nn.Linear(combined_dim_high, combined_dim_high)
        self.out_layer_v_high = nn.Linear(combined_dim_high, output_dim)
        self.proj1_a_high    = nn.Linear(combined_dim_high, combined_dim_high)
        self.proj2_a_high    = nn.Linear(combined_dim_high, combined_dim_high)
        self.out_layer_a_high = nn.Linear(combined_dim_high, output_dim)

        # [7] Fusion projectors
        self.projector_l = nn.Linear(self.d_l, self.d_l)
        self.projector_v = nn.Linear(self.d_v, self.d_v)
        self.projector_a = nn.Linear(self.d_a, self.d_a)
        self.projector_c = nn.Linear(3 * self.d_l, 3 * self.d_l)

        # [8] Final prediction head
        self.proj1     = nn.Linear(combined_dim, combined_dim)
        self.proj2     = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)

    def get_network(self, self_type='l', layers=-1):
        if self_type in ['l', 'al', 'vl']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type in ['a', 'la', 'va']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ['v', 'lv', 'av']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        elif self_type == 'l_mem':
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = self.d_a, self.attn_dropout
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = self.d_v, self.attn_dropout
        else:
            raise ValueError(f"Unknown network type: {self_type}")
        return TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=self.num_heads,
            layers=max(self.layers, layers),
            attn_dropout=attn_dropout,
            relu_dropout=self.relu_dropout,
            res_dropout=self.res_dropout,
            embed_dropout=self.embed_dropout,
            attn_mask=self.attn_mask
        )

    def forward(self, text, audio, video):
        # --- Feature extraction ---
        if self.use_bert:
            text = self.text_model(text)
        x_l = F.dropout(text.transpose(1, 2), p=self.text_dropout, training=self.training)
        x_a = audio.transpose(1, 2)
        x_v = video.transpose(1, 2)

        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)

        # (seq_len, batch, dim)
        proj_x_l = proj_x_l.permute(2, 0, 1)
        proj_x_v = proj_x_v.permute(2, 0, 1)
        proj_x_a = proj_x_a.permute(2, 0, 1)

        # --- Disentanglement: specific + shared encoders ---
        s_l = self.encoder_s_l(proj_x_l)
        s_v = self.encoder_s_v(proj_x_v)
        s_a = self.encoder_s_a(proj_x_a)

        c_l = self.encoder_c(proj_x_l)
        c_v = self.encoder_c(proj_x_v)
        c_a = self.encoder_c(proj_x_a)

        # --- SDA: noise-filtered shared representations ---
        c_adp_l = self.sda_l(proj_x_l)
        c_adp_v = self.sda_v(proj_x_v)
        c_adp_a = self.sda_a(proj_x_a)

        c_l = self.lin(torch.cat((c_l, c_adp_l), dim=-1))
        c_v = self.lin(torch.cat((c_v, c_adp_v), dim=-1))
        c_a = self.lin(torch.cat((c_a, c_adp_a), dim=-1))

        # (batch, dim, seq_len)
        s_l = s_l.permute(1, 2, 0)
        s_v = s_v.permute(1, 2, 0)
        s_a = s_a.permute(1, 2, 0)
        c_l = c_l.permute(1, 2, 0)
        c_v = c_v.permute(1, 2, 0)
        c_a = c_a.permute(1, 2, 0)
        c_list = [c_l, c_v, c_a]

        c_l_sim = self.align_c_l(c_l.contiguous().view(x_l.size(0), -1))
        c_v_sim = self.align_c_v(c_v.contiguous().view(x_l.size(0), -1))
        c_a_sim = self.align_c_a(c_a.contiguous().view(x_l.size(0), -1))

        # Reconstruction
        recon_l = self.decoder_l(torch.cat([s_l, c_list[0]], dim=1))
        recon_v = self.decoder_v(torch.cat([s_v, c_list[1]], dim=1))
        recon_a = self.decoder_a(torch.cat([s_a, c_list[2]], dim=1))

        recon_l = recon_l.permute(2, 0, 1)
        recon_v = recon_v.permute(2, 0, 1)
        recon_a = recon_a.permute(2, 0, 1)

        s_l_r = self.encoder_s_l(recon_l).permute(1, 2, 0)
        s_v_r = self.encoder_s_v(recon_v).permute(1, 2, 0)
        s_a_r = self.encoder_s_a(recon_a).permute(1, 2, 0)

        # (seq_len, batch, dim) for attention ops
        s_l = s_l.permute(2, 0, 1)
        s_v = s_v.permute(2, 0, 1)
        s_a = s_a.permute(2, 0, 1)
        c_l = c_l.permute(2, 0, 1)
        c_v = c_v.permute(2, 0, 1)
        c_a = c_a.permute(2, 0, 1)

        # Shared self-attention
        c_l_att = self.self_attentions_c_l(c_l)
        if isinstance(c_l_att, tuple): c_l_att = c_l_att[0]
        c_l_att = c_l_att[-1]
        c_v_att = self.self_attentions_c_v(c_v)
        if isinstance(c_v_att, tuple): c_v_att = c_v_att[0]
        c_v_att = c_v_att[-1]
        c_a_att = self.self_attentions_c_a(c_a)
        if isinstance(c_a_att, tuple): c_a_att = c_a_att[0]
        c_a_att = c_a_att[-1]
        c_fusion = torch.cat([c_l_att, c_v_att, c_a_att], dim=1)

        # --- FGA: Feature-Guided Alignment ---
        (cross_l, cross_v, cross_a,
         g_l, g_v, g_a,
         attn_l, attn_v, attn_a,
         shared_anchor) = self.fga(
            shared_feat_l=c_l.mean(dim=0),
            shared_feat_v=c_v.mean(dim=0),
            shared_feat_a=c_a.mean(dim=0),
            spec_feat_l=s_l.permute(1, 0, 2),
            spec_feat_v=s_v.permute(1, 0, 2),
            spec_feat_a=s_a.permute(1, 0, 2)
        )

        # --- DPF: Dynamic Perspective Fusion ---
        dyn_l, dyn_v, dyn_a, w_l, w_v, w_a = self.dpf(
            cross_l, cross_v, cross_a, shared_anchor
        )

        # Dynamic prediction targets (L_dp)
        dynamic_loss_l = self.dyn_proj(dyn_l).mean(dim=1)
        dynamic_loss_v = self.dyn_proj(dyn_v).mean(dim=1)
        dynamic_loss_a = self.dyn_proj(dyn_a).mean(dim=1)

        dyn_l_seq = dyn_l.permute(1, 0, 2)
        dyn_v_seq = dyn_v.permute(1, 0, 2)
        dyn_a_seq = dyn_a.permute(1, 0, 2)

        # Contrastive loss (L_nce)
        nce_loss = (self.cpc_loss(c_l, dyn_l_seq) +
                    self.cpc_loss(c_v, dyn_v_seq) +
                    self.cpc_loss(c_a, dyn_a_seq))

        # Aggregate dynamic features
        last_h_l = dyn_l.mean(dim=0)
        last_h_v = dyn_v.mean(dim=0)
        last_h_a = dyn_a.mean(dim=0)

        c_fusion = torch.sigmoid(self.projector_c(c_fusion))
        last_hs  = torch.cat([last_h_l, last_h_v, last_h_a, c_fusion], dim=1)

        # Final prediction
        last_hs_proj = self.proj2(
            F.dropout(F.relu(self.proj1(last_hs), inplace=True),
                      p=self.output_dropout, training=self.training)
        )
        last_hs_proj += last_hs
        output = self.out_layer(last_hs_proj)

        return {
            'origin_l':       proj_x_l,
            'origin_v':       proj_x_v,
            'origin_a':       proj_x_a,
            's_l':            s_l,
            's_v':            s_v,
            's_a':            s_a,
            'c_l':            c_l,
            'c_v':            c_v,
            'c_a':            c_a,
            's_l_r':          s_l_r,
            's_v_r':          s_v_r,
            's_a_r':          s_a_r,
            'recon_l':        recon_l,
            'recon_v':        recon_v,
            'recon_a':        recon_a,
            'c_l_sim':        c_l_sim,
            'c_v_sim':        c_v_sim,
            'c_a_sim':        c_a_sim,
            'w_l':            w_l,
            'w_v':            w_v,
            'w_a':            w_a,
            'dynamic_loss_l': dynamic_loss_l,
            'dynamic_loss_v': dynamic_loss_v,
            'dynamic_loss_a': dynamic_loss_a,
            'dyn_l':          dyn_l_seq,
            'dyn_v':          dyn_v_seq,
            'dyn_a':          dyn_a_seq,
            'output_logit':   output,
            'nce_loss':       nce_loss,
        }
