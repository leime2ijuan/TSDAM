import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleEncoder(nn.Module):
    """改进的多尺度特征编码器"""
    def __init__(self, input_dim, hidden_dim, num_scales=2):
        # num_scales参数含义：表示特征提取的尺度数量。
        # 2 表示“粗/细”两种尺度，3 则可进一步分为“粗/中/细”。
#         粗粒度特征：捕获数据的整体模式和长期趋势
# 中粒度特征：捕获数据的局部模式和中期变化
# 细粒度特征：捕获数据的细节信息和短期波动
        super().__init__()
        self.input_dim = input_dim          # 输入特征维度，例如6（5个天气特征加1个电力数据特征）
        self.hidden_dim = hidden_dim        # 隐藏层维度，决定了特征表示的维度大小
        self.scales = num_scales            # 多尺度分支的数量，每个分支独立抽取不同粒度的特征
        
        # 输入投影层：将输入特征映射到隐藏层维度空间
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # 多尺度特征提取层：为每个尺度创建独立的特征提取器
        # 每个提取器由线性层、层归一化、ReLU激活和Dropout组成
        # 每个尺度的 线性层 都有自己的 权重矩阵 和 偏置项。
        # 每个尺度的 层归一化 层（nn.LayerNorm）也有自己的 缩放参数（gamma）和 偏移参数（beta）
        # 共享ReLU 激活函数、Dropout 
        self.feature_extractors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ) for _ in range(num_scales)
        ])
        
        # 特征融合层：将所有尺度分支的输出特征拼接后进行融合
        # 输入维度是 hidden_dim * num_scales（所有分支特征拼接后的维度）
        # 输出维度映射回 hidden_dim，以便后续处理
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * num_scales, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
    def forward(self, x):
        # 输入投影：将输入特征映射到隐藏层维度
        if x.size(-1) != self.hidden_dim:
            x = self.input_projection(x)
        
        # 多尺度特征提取：每个尺度的提取器处理相同的输入，捕获不同粒度的特征
        multi_scale_features = []
        for extractor in self.feature_extractors:
            features = extractor(x)
            multi_scale_features.append(features)
        
        # 特征拼接：将所有尺度的特征在最后一个维度上拼接
        concatenated = torch.cat(multi_scale_features, dim=-1)
        
        # 特征融合：通过线性层将拼接后的高维特征映射回隐藏层维度
        fused_features = self.fusion(concatenated)
        
        return fused_features

class TimeSeriesEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_domains=7, num_layers=2, dropout=0.3):
        super().__init__()
        self.input_dim = input_dim
        # 确保 hidden_dim 能被 num_heads (4) 整除，以适应多头注意力机制的要求
        self.hidden_dim = (hidden_dim // 4) * 4
        self.num_domains = num_domains
        
        # 多尺度特征提取 - 关键修改：直接使用完整的hidden_dim
        self.multi_scale_encoder = MultiScaleEncoder(
            input_dim=input_dim,
            hidden_dim=64  # 这里硬编码为64，可能需要与hidden_dim保持一致或调整
        )
        
        # 时间注意力层：使用多头注意力机制捕获时间序列中的依赖关系
        # embed_dim：输入序列的特征维度
        # num_heads：注意力头的数量，将特征维度分成多个头并行处理
        # dropout：应用于注意力权重的dropout率
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=4,
            dropout=dropout
        )
        
        # 改进的贝叶斯域自适应模块：处理不同领域之间的差异
        # domain_mu：每个域的均值参数，用于贝叶斯采样
        # domain_logvar：每个域的对数方差参数，用于贝叶斯采样
        # domain_importance：每个域的重要性权重，用于平衡不同域的影响
        self.domain_mu = nn.Parameter(torch.zeros(num_domains, self.hidden_dim))
        self.domain_logvar = nn.Parameter(torch.zeros(num_domains, self.hidden_dim))
        self.domain_importance = nn.Parameter(torch.ones(num_domains))
        
        # 特征整合层：将多尺度特征和时间注意力特征融合
        self.feature_fusion = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(self.hidden_dim)
        )
        
        # 添加域适应基础网络：对贝叶斯采样后的特征进行进一步变换
        self.domain_adapter_base = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def bayesian_domain_adapt(self, features, domain_idx=None):
        # 贝叶斯域自适应方法：基于贝叶斯原理对特征进行域适应
        if domain_idx is not None:
            # 单域适应模式：使用指定域的参数
            mu = self.domain_mu[domain_idx]
            logvar = self.domain_logvar[domain_idx]
            importance = F.softplus(self.domain_importance[domain_idx])
            
            # 从高斯分布中采样参数：mu + eps * std
            std = torch.exp(0.5 * logvar)  # 从对数方差计算标准差
            eps = torch.randn_like(std)    # 生成与标准差相同形状的随机噪声
            domain_params = mu + eps * std  # 采样得到域特定参数
            
            # 使用采样的参数调整特征，并应用域重要性权重
            adapted_features = features * (domain_params * importance.unsqueeze(-1))
        else:
            # 混合域适应模式：融合所有域的信息
            domain_weights = F.softmax(self.domain_importance, dim=0)  # 将重要性转换为权重
            
            # 计算混合域的均值和对数方差
            mixed_mu = torch.sum(self.domain_mu * domain_weights.unsqueeze(1), dim=0)
            mixed_logvar = torch.sum(self.domain_logvar * domain_weights.unsqueeze(1), dim=0)
            
            # 从混合高斯分布中采样参数
            std = torch.exp(0.5 * mixed_logvar)
            eps = torch.randn_like(std)
            domain_params = mixed_mu + eps * std
            
            # 使用混合参数调整特征
            adapted_features = features * domain_params
        
        # 通过基础网络进一步处理适应后的特征
        return self.domain_adapter_base(adapted_features)
    
    def forward(self, x, domain_idx=None):
        batch_size, num_buildings, seq_len, _ = x.shape
        x_reshaped = x.view(batch_size * num_buildings, seq_len, self.input_dim)
        
        # 1. 多尺度特征提取：获取融合后的多尺度特征
        adjusted_features = self.multi_scale_encoder(x_reshaped)
        
        # 2. 时间注意力：应用自注意力机制捕获时间依赖关系
        # 注意：输入需要调整为 (seq_len, batch, feature) 的格式
        attended_features, _ = self.temporal_attention(
            adjusted_features.transpose(0, 1),  # 查询
            adjusted_features.transpose(0, 1),  # 键
            adjusted_features.transpose(0, 1)   # 值
        )
        # 将输出转回 (batch, seq_len, feature) 的格式
        attended_features = attended_features.transpose(0, 1)
        
        # 3. 特征融合：将多尺度特征和时间注意力特征拼接并融合
        fused_features = self.feature_fusion(
            torch.cat([adjusted_features, attended_features], dim=-1)
        )
        
        # 4. 贝叶斯域适应：对每个时间步的特征应用域适应
        adapted_features = []
        for t in range(seq_len):
            t_feat = fused_features[:, t, :]  # 获取当前时间步的特征
            t_adapted = self.bayesian_domain_adapt(t_feat, domain_idx)  # 应用域适应
            adapted_features.append(t_adapted)
        # fused_features_reshaped = fused_features.view(-1, self.hidden_dim)  # [batch*seq_len, hidden_dim]
        # adapted_features_reshaped = self.bayesian_domain_adapt(fused_features_reshaped, domain_idx)
        # adapted_features = adapted_features_reshaped.view(batch_size, seq_len, self.hidden_dim)
        # 将各时间步的特征重新组合成序列
        adapted_features = torch.stack(adapted_features, dim=1)
        
        # 恢复原始的批次和建筑物维度
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

class DomainDiscriminator(nn.Module):
    def __init__(self, feature_dim, hidden_dim=64, dropout=0.3):
        super(DomainDiscriminator, self).__init__()
        self.feature_dim = feature_dim
        
        self.simple_model = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)  # 二分类，输出一个值
        )
        
    def forward(self, x):
        return self.simple_model(x)