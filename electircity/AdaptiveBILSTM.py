import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Function

class MultiScaleEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_scales=2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.scales = num_scales
        
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        self.feature_extractors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ) for _ in range(num_scales)
        ])
        
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * num_scales, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
    def forward(self, x):
        if x.size(-1) != self.hidden_dim:
            x = self.input_projection(x)
        
        multi_scale_features = []
        for extractor in self.feature_extractors:
            features = extractor(x)
            multi_scale_features.append(features)
        
        concatenated = torch.cat(multi_scale_features, dim=-1)
        fused_features = self.fusion(concatenated)
        
        return fused_features


class TemporalDependency(nn.Module):
    def __init__(self, hidden_dim=64, weather_dim=5, num_heads=4, dropout=0.3):
        super().__init__()
        self.hidden_dim = (hidden_dim // num_heads) * num_heads
        self.num_heads = num_heads
        self.head_dim = self.hidden_dim // num_heads

        self.q_conv = nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1, groups=num_heads)
        self.k_conv = nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1, groups=num_heads)
        self.v_conv = nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1, groups=num_heads)

        self.weather_fc = nn.Linear(weather_dim, self.hidden_dim)
        
        self.linear = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, F_scale, weather):
        B, T, _ = F_scale.size()

        Q = F_scale.permute(0, 2, 1)
        K = F_scale.permute(0, 2, 1)
        V = F_scale.permute(0, 2, 1)

        Q_conv = self.q_conv(Q).permute(0, 2, 1)
        K_conv = self.k_conv(K).permute(0, 2, 1)
        V_conv = self.v_conv(V).permute(0, 2, 1)

        W_embed = self.weather_fc(weather)
        sim = torch.bmm(W_embed, W_embed.transpose(1, 2))
        g_bias = torch.tanh(sim / math.sqrt(self.hidden_dim))

        scale = math.sqrt(self.head_dim)
        attn_logits = torch.bmm(Q_conv, K_conv.transpose(1, 2)) / scale
        attn_logits = attn_logits + g_bias
        attn_weights = torch.softmax(attn_logits, dim=-1)
        attn_output = torch.bmm(attn_weights, V_conv)

        output = self.linear(attn_output)
        output = self.dropout(output)
        return output


class TimeSeriesEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_domains=7, num_layers=2, dropout=0.3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = (hidden_dim // 4) * 4
        self.num_domains = num_domains
        
        self.multi_scale_encoder = MultiScaleEncoder(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim
        )
        
        self.temporal_dependency = TemporalDependency(
            hidden_dim=self.hidden_dim,
            weather_dim=input_dim - 1,
            num_heads=4,
            dropout=dropout
        )
        
        self.domain_mu = nn.Parameter(torch.zeros(num_domains, self.hidden_dim))
        self.domain_logvar = nn.Parameter(torch.zeros(num_domains, self.hidden_dim))
        self.domain_importance = nn.Parameter(torch.ones(num_domains))
        
        self.feature_fusion = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(self.hidden_dim)
        )
        
        self.domain_adapter_base = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def gaussian_domain_adapt(self, features, domain_idx=None):
        if domain_idx is not None:
            mu = self.domain_mu[domain_idx]
            logvar = self.domain_logvar[domain_idx]
            importance = F.softplus(self.domain_importance[domain_idx])
            
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            domain_params = mu + eps * std
            
            adapted_features = features * (domain_params * importance.unsqueeze(-1))
        else:
            domain_weights = F.softmax(self.domain_importance, dim=0)
            mixed_mu = torch.sum(self.domain_mu * domain_weights.unsqueeze(1), dim=0)
            mixed_logvar = torch.sum(self.domain_logvar * domain_weights.unsqueeze(1), dim=0)
            
            std = torch.exp(0.5 * mixed_logvar)
            eps = torch.randn_like(std)
            domain_params = mixed_mu + eps * std
            
            adapted_features = features * domain_params
        
        return self.domain_adapter_base(adapted_features)
    
    def forward(self, x, domain_idx=None):
        batch_size, num_buildings, seq_len, _ = x.shape
        
        energy = x[..., 0].unsqueeze(-1)
        weather = x[..., 1:]
    
        x_reshaped = x.view(batch_size * num_buildings, seq_len, self.input_dim)
        weather_reshaped = weather.view(batch_size * num_buildings, seq_len, self.input_dim - 1)
    
        adjusted_features = self.multi_scale_encoder(x_reshaped)
    
        attended_features = self.temporal_dependency(adjusted_features, weather_reshaped)
    
        fused_features = self.feature_fusion(
            torch.cat([adjusted_features, attended_features], dim=-1)
        )
    
        adapted_features = []
        for t in range(seq_len):
            t_feat = fused_features[:, t, :]
            t_adapted = self.gaussian_domain_adapt(t_feat, domain_idx)
            adapted_features.append(t_adapted)
        adapted_features = torch.stack(adapted_features, dim=1)
    
        return adapted_features.view(batch_size, num_buildings, seq_len, self.hidden_dim)


class BiLSTMPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, category_dim, forecast_horizon, num_buildings, num_layers=2, dropout=0.3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.forecast_horizon = forecast_horizon
        self.num_buildings = num_buildings
        
        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.category_encoder = nn.Sequential(
            nn.Linear(category_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim*2,
            num_heads=4,
            dropout=dropout
        )
        
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim*2 + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, forecast_horizon),
            nn.Sigmoid()
        )
        
    def forward(self, x, category):
        batch_size, num_buildings, seq_len, _ = x.shape
        
        category_embed = self.category_encoder(category)
        
        all_predictions = []
        for b in range(num_buildings):
            building_data = x[:, b, :, :]
            
            lstm_out, _ = self.bilstm(building_data)
            
            lstm_out_t = lstm_out.transpose(0, 1)
            attended_out, _ = self.temporal_attention(lstm_out_t, lstm_out_t, lstm_out_t)
            attended_out = attended_out.transpose(0, 1)
            
            final_hidden = attended_out[:, -1, :]
            
            combined = torch.cat([final_hidden, category_embed], dim=1)
            pred = self.prediction_head(combined).unsqueeze(1)
            
            all_predictions.append(pred)
            
        return torch.cat(all_predictions, dim=1)


class AdaptiveBiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, category_dim, forecast_horizon, num_buildings, num_domains=7, num_layers=2, dropout=0.3):
        super(AdaptiveBiLSTM, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.forecast_horizon = forecast_horizon
        self.num_buildings = num_buildings
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.time_series_encoder = TimeSeriesEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_domains=num_domains,
            num_layers=num_layers,
            dropout=dropout
        )
        
        self.bilstm_predictor = BiLSTMPredictor(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            category_dim=category_dim,
            forecast_horizon=forecast_horizon,
            num_buildings=num_buildings,
            num_layers=num_layers,
            dropout=dropout
        )
    
    def forward(self, x, category, domain_idx=None):
        time_features = self.time_series_encoder(x, domain_idx)
        predictions = self.bilstm_predictor(time_features, category)
        return predictions


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

def grad_reverse(x, alpha=1.0):
    return GradientReversalFunction.apply(x, alpha)


class DomainDiscriminator(nn.Module):
    def __init__(self, feature_dim, hidden_dim=64, dropout=0.3):
        super(DomainDiscriminator, self).__init__()
        self.feature_dim = feature_dim
        
        self.simple_model = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x, alpha=1.0):
        reversed_x = grad_reverse(x, alpha)
        return self.simple_model(reversed_x)
