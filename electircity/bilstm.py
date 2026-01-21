import torch
import torch.nn as nn
import torch.nn.functional as F

class BaselineBiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, forecast_horizon, num_buildings, num_layers=1, dropout=0.1, output_var=False):
        super(BaselineBiLSTM, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.forecast_horizon = forecast_horizon
        self.num_layers = num_layers
        self.num_buildings = num_buildings
        self.output_var = output_var  # 新增参数，控制是否输出方差
        
        # BiLSTM层
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 输出层 - 预测均值
        self.mean_output = nn.Linear(hidden_dim * 2, forecast_horizon)
        
        # 输出层 - 预测方差
        if output_var:
            self.var_output = nn.Linear(hidden_dim * 2, forecast_horizon)
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, category_onehot=None):
        """
        前向传播
        
        参数:
        - x: 输入序列 [batch_size, num_buildings, sequence_length, input_dim]
        - category_onehot: 类别one-hot编码 (可选)
        
        返回:
        - mean: 预测均值，如果output_var=True，则返回(mean, var)
        """
        batch_size, num_buildings, seq_len, _ = x.size()
        
        # 重塑输入以处理每个建筑
        x = x.view(batch_size * num_buildings, seq_len, self.input_dim)
        
        # BiLSTM处理
        lstm_out, _ = self.lstm(x)
        
        # 取最后一个时间步的输出
        last_out = lstm_out[:, -1, :]
        
        # 应用dropout
        last_out = self.dropout(last_out)
        
        # 预测均值
        mean = self.mean_output(last_out)
        
        # 重塑回原始批次大小和建筑数量
        mean = mean.view(batch_size, num_buildings, self.forecast_horizon)
        
        # 如果需要输出方差
        if self.output_var:
            # 预测方差
            log_var = self.var_output(last_out)
            var = torch.exp(log_var)
            var = var.view(batch_size, num_buildings, self.forecast_horizon)
            return mean, var
        else:
            # 只返回均值
            return mean