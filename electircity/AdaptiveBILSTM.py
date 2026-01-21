import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Function

class MultiScaleEncoder(nn.Module):
    """改进的多尺度特征编码器"""
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
    """修复后的时间依赖模块"""
    def __init__(self, hidden_dim=64, weather_dim=5, num_heads=4, dropout=0.3):
        super().__init__()
        # 确保hidden_dim能被num_heads整除
        self.hidden_dim = (hidden_dim // num_heads) * num_heads
        self.num_heads = num_heads
        self.head_dim = self.hidden_dim // num_heads

        # 使用1D卷积进行局部特征提取
        self.q_conv = nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1, groups=num_heads)
        self.k_conv = nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1, groups=num_heads)
        self.v_conv = nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1, groups=num_heads)

        # 天气特征处理
        self.weather_fc = nn.Linear(weather_dim, self.hidden_dim)
        
        # 输出层
        self.linear = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, F_scale, weather):
        B, T, _ = F_scale.size()

        # 转换维度用于卷积 [B, D, T]
        Q = F_scale.permute(0, 2, 1)
        K = F_scale.permute(0, 2, 1)
        V = F_scale.permute(0, 2, 1)

        # 应用卷积并恢复维度 [B, T, D]
        Q_conv = self.q_conv(Q).permute(0, 2, 1)
        K_conv = self.k_conv(K).permute(0, 2, 1)
        V_conv = self.v_conv(V).permute(0, 2, 1)

        # 天气偏置计算 - 修复这里，不需要额外的bias_fc层
        W_embed = self.weather_fc(weather)  # [B, T, hidden_dim]
        sim = torch.bmm(W_embed, W_embed.transpose(1, 2))  # [B, T, T]
        g_bias = torch.tanh(sim / math.sqrt(self.hidden_dim))  # 直接计算，不使用bias_fc

        # 注意力计算
        scale = math.sqrt(self.head_dim)  # 使用head_dim而不是hidden_dim/num_heads
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
        # 确保 hidden_dim 能被 num_heads (4) 整除
        self.hidden_dim = (hidden_dim // 4) * 4
        self.num_domains = num_domains
        
        # 多尺度特征提取 - 使用统一的hidden_dim
        self.multi_scale_encoder = MultiScaleEncoder(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim  # 修复：使用self.hidden_dim而不是硬编码64
        )
        
        # 时间注意力层 - 使用修复后的TemporalDependency
        self.temporal_dependency = TemporalDependency(
            hidden_dim=self.hidden_dim,
            weather_dim=input_dim - 1,
            num_heads=4,
            dropout=dropout
        )
        
        # 贝叶斯域自适应参数
        self.domain_mu = nn.Parameter(torch.zeros(num_domains, self.hidden_dim))
        self.domain_logvar = nn.Parameter(torch.zeros(num_domains, self.hidden_dim))
        self.domain_importance = nn.Parameter(torch.ones(num_domains))
        
        # 特征融合层
        self.feature_fusion = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(self.hidden_dim)
        )
        
        # 域适应基础网络
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
        
        # 拆分能耗和天气
        energy = x[..., 0].unsqueeze(-1)     # 电耗 [B,N,T,1]
        weather = x[..., 1:]                 # 天气 [B,N,T,5]
    
        # 调整形状以便编码
        x_reshaped = x.view(batch_size * num_buildings, seq_len, self.input_dim)
        weather_reshaped = weather.view(batch_size * num_buildings, seq_len, self.input_dim - 1)
    
        # 1. 多尺度特征提取
        adjusted_features = self.multi_scale_encoder(x_reshaped)
    
        # 2. 时序依赖模块（卷积+天气偏置）
        attended_features = self.temporal_dependency(adjusted_features, weather_reshaped)
    
        # 3. 特征融合
        fused_features = self.feature_fusion(
            torch.cat([adjusted_features, attended_features], dim=-1)
        )
    
        # 4. 贝叶斯域适应
        adapted_features = []
        for t in range(seq_len):
            t_feat = fused_features[:, t, :]
            t_adapted = self.gaussian_domain_adapt(t_feat, domain_idx)
            adapted_features.append(t_adapted)
        adapted_features = torch.stack(adapted_features, dim=1)
    
        return adapted_features.view(batch_size, num_buildings, seq_len, self.hidden_dim)

class BiLSTMPredictor(nn.Module):
    """
    基于双向LSTM和注意力机制的多建筑物能源预测模型
    
    该模型通过以下方式工作：
    1. 使用双向LSTM提取时序特征
    2. 应用多头注意力机制增强特征表示
    3. 融合类别特征以捕获建筑物特定属性
    4. 为每个建筑物生成未来能耗预测
    """
    
    def __init__(self, input_dim, hidden_dim, category_dim, forecast_horizon, num_buildings, num_layers=2, dropout=0.3):
        """
        初始化模型参数和层结构
        
        参数:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            category_dim: 类别特征维度
            forecast_horizon: 预测时间步长
            num_buildings: 建筑物数量
            num_layers: LSTM层数
            dropout: Dropout比率
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.forecast_horizon = forecast_horizon
        self.num_buildings = num_buildings
        
        # ===== 特征提取层 =====
        # 双向LSTM层 - 捕获时序数据中的双向依赖关系
        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,  # 输入格式为(batch, seq, feature)
            dropout=dropout if num_layers > 1 else 0,  # 仅在多层LSTM时应用dropout
            bidirectional=True  # 使用双向LSTM
        )
        
        # ===== 类别特征编码层 =====
        # 将类别特征映射到高维空间，增强模型表达能力
        self.category_encoder = nn.Sequential(
            nn.Linear(category_dim, hidden_dim),
            nn.ReLU(),  # 引入非线性
            nn.Dropout(dropout),  # 防止过拟合
            nn.Linear(hidden_dim, hidden_dim)  # 进一步特征变换
        )
        
        # ===== 注意力机制层 =====
        # 时序注意力 - 自适应地关注序列中的重要部分
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim*2,  # 双向LSTM输出维度翻倍
            num_heads=4,  # 使用4个头并行计算注意力
            dropout=dropout
        )
        
        # ===== 预测输出层 =====
        # 融合时序特征和类别特征，生成最终预测
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim*2 + hidden_dim, hidden_dim),  # 拼接双向LSTM和类别特征
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),  # 稳定训练过程
            nn.Linear(hidden_dim, forecast_horizon),  # 输出预测时间步长
            nn.Sigmoid()  # 将输出归一化到[0,1]范围
        )
        
    def forward(self, x, category):
        """
        前向传播过程
        
        参数:
            x: 输入时序数据 (batch_size, num_buildings, seq_len, input_dim)
            category: 建筑物类别特征 (batch_size, category_dim)
            
        返回:
            predictions: 预测结果 (batch_size, num_buildings, forecast_horizon)
        """
        batch_size, num_buildings, seq_len, _ = x.shape
        
        # 编码类别特征
        category_embed = self.category_encoder(category)  # (batch_size, hidden_dim)
        
        all_predictions = []
        # 逐个建筑物处理
        for b in range(num_buildings):
            # 提取单个建筑物的时序数据
            building_data = x[:, b, :, :]  # (batch_size, seq_len, input_dim)
            
            # ===== 时序特征提取 =====
            # 通过双向LSTM提取时序特征
            lstm_out, _ = self.bilstm(building_data)  # (batch_size, seq_len, hidden_dim*2)
            
            # ===== 应用注意力机制 =====
            # 调整维度以适应注意力层输入要求
            # MultiheadAttention期望输入格式为(seq_len, batch_size, embed_dim)
            lstm_out_t = lstm_out.transpose(0, 1)  # (seq_len, batch_size, hidden_dim*2)
            
            # 应用多头注意力机制
            attended_out, _ = self.temporal_attention(
                lstm_out_t,  # 查询序列
                lstm_out_t,  # 键序列
                lstm_out_t   # 值序列
            )  # (seq_len, batch_size, hidden_dim*2)
            
            # 恢复原始维度顺序
            attended_out = attended_out.transpose(0, 1)  # (batch_size, seq_len, hidden_dim*2)
            
            # 获取序列最后一个时间步的特征表示
            final_hidden = attended_out[:, -1, :]  # (batch_size, hidden_dim*2)
            
            # ===== 特征融合与预测 =====
            # 拼接时序特征和类别特征
            combined = torch.cat([final_hidden, category_embed], dim=1)  # (batch_size, hidden_dim*2 + hidden_dim)
            
            # 通过预测头生成预测结果
            pred = self.prediction_head(combined).unsqueeze(1)  # (batch_size, 1, forecast_horizon)
            
            all_predictions.append(pred)
            
        # 合并所有建筑物的预测结果
        return torch.cat(all_predictions, dim=1)  # (batch_size, num_buildings, forecast_horizon)

# 修改主模型以适应新的预测架构和名称
class AdaptiveBiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, category_dim, forecast_horizon, num_buildings, num_domains=7, num_layers=2, dropout=0.3):
        super(AdaptiveBiLSTM, self).__init__()
        
        # 保存维度信息用于外部访问
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
        # 提取时间特征并应用贝叶斯域自适应
        time_features = self.time_series_encoder(x, domain_idx)
        
        # 预测，直接返回预测值
        predictions = self.bilstm_predictor(time_features, category)
        
        return predictions
# 梯度反转层 - 域对抗训练的核心组件
class GradientReversalFunction(Function):
    """
    梯度反转层 - 在反向传播时反转梯度方向，实现域对抗训练
    前向传播：直接传递输入
    反向传播：将梯度乘以负的alpha值，实现梯度反转
    """
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha  # 存储alpha参数用于反向传播
        return x.view_as(x)  # 前向传播不改变输入
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None  # 反向传播时反转梯度方向

def grad_reverse(x, alpha=1.0):
    """梯度反转函数的包装，便于调用"""
    return GradientReversalFunction.apply(x, alpha)

# 域判别器 - 用于区分源域和目标域特征
class DomainDiscriminator(nn.Module):
    """
    域判别器 - 用于区分源域和目标域特征
    
    Args:
        feature_dim: 输入特征维度
        hidden_dim: 隐藏层维度
        dropout: Dropout比率
    """
    def __init__(self, feature_dim, hidden_dim=64, dropout=0.3):
        super(DomainDiscriminator, self).__init__()
        self.feature_dim = feature_dim
        
        # 简单判别器模型
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
        # 应用梯度反转
        reversed_x = grad_reverse(x, alpha)
        return self.simple_model(reversed_x)
