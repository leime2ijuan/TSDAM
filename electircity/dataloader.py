
"""
dataloader.py

时间序列数据加载器 - 支持混合数据稀缺场景
- 训练建筑使用完整数据
- 目标建筑使用稀缺数据 (mild/heavy/extreme)
- 支持训练/测试时间划分
- 支持CC类别特殊处理
    """
import pandas as pd
import numpy as np
import os
import json
import torch
import random
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from typing import Tuple, List, Dict, Optional

# ===== 在所有 import 之后，class ChronosDataset 之前添加 =====

def custom_collate_fn(batch):
    """
    自定义批次合并函数，处理混合建筑数量的数据
    
    输入格式：
        batch: list of (features, targets, category, category_onehot)
        - features: [num_buildings, seq_len, input_dim]
        - targets: [num_buildings, forecast_horizon]
    
    输出格式：
        - features_batch: [total_buildings, 1, seq_len, input_dim]
        - targets_batch: [total_buildings, 1, forecast_horizon]
        - category_batch: str (所有样本共享同一类别)
        - category_onehot_batch: [total_buildings, num_categories]
    
    逻辑：
        将不同样本中的所有建筑展平到 batch 维度，
        每个建筑作为一个独立的样本（建筑维度=1）
    """
    features_list = []
    targets_list = []
    categories_list = []
    category_onehot_list = []
    
    for item in batch:
        features, targets, category, category_onehot = item
        
        # 获取当前样本的建筑数量
        num_buildings = features.shape[0]
        
        # 将每个建筑作为独立样本
        for b in range(num_buildings):
            # 提取第 b 个建筑的数据，保持维度 [1, seq_len, features]
            features_list.append(features[b:b+1, :, :])  # [1, seq_len, input_dim]
            targets_list.append(targets[b:b+1, :])        # [1, forecast_horizon]
            
            # 类别信息（所有建筑共享）
            categories_list.append(category)
            category_onehot_list.append(category_onehot)
    
    # 合并所有建筑
    # features_batch: [total_buildings, seq_len, input_dim]
    features_batch = torch.cat(features_list, dim=0)
    
    # targets_batch: [total_buildings, forecast_horizon]
    targets_batch = torch.cat(targets_list, dim=0)
    
    # 添加建筑维度（统一为1）以保持原始数据格式
    # features_batch: [total_buildings, 1, seq_len, input_dim]
    features_batch = features_batch.unsqueeze(1)
    
    # targets_batch: [total_buildings, 1, forecast_horizon]
    targets_batch = targets_batch.unsqueeze(1)
    
    # 类别信息（第一个样本的类别，因为同一批次都是同类别）
    category_batch = categories_list[0]
    
    # category_onehot: [total_buildings, num_categories]
    category_onehot_batch = torch.stack(category_onehot_list, dim=0)
    
    return features_batch, targets_batch, category_batch, category_onehot_batch

class ChronosDataset(Dataset):
    """
    Chronos时间序列数据集
    
    支持多建筑、多特征的时间序列数据，包含电力消耗和天气数据。
    
    数据格式:
        - 输入: [buildings, sequence_length, features] 
        - 目标: [buildings, forecast_horizon]
        - 特征: [电力, 气温, 露点温度, 气压, 风向, 风速]
    
    参数:
        electricity_data: 电力数据 DataFrame (index: timestamp, columns: buildings)
        weather_data: 天气数据字典 {station_name: DataFrame}
        buildings: 建筑ID列表
        building_to_weather: 建筑到天气站的映射字典
        building_to_category: 建筑到类别的映射字典
        categories: 所有类别列表
        sequence_length: 输入序列长度（小时）
        forecast_horizon: 预测步长（小时）
        handle_missing: 缺失值处理方式 ('forward_fill' 或 'zero_fill')
        time_range: 时间范围限制 ('all', 'first_80_percent', 'last_20_percent')
    """
    
    def __init__(self, 
                 electricity_data: pd.DataFrame,
                 weather_data: Dict[str, pd.DataFrame],
                 buildings: List[str],
                 building_to_weather: Dict[str, str],
                 building_to_category: Dict[str, str],
                 categories: List[str],
                 sequence_length: int = 24,
                 forecast_horizon: int = 24,
                 handle_missing: str = 'forward_fill',
                 time_range: str = 'all'):
        
        self.electricity_data = electricity_data
        self.weather_data = weather_data
        self.buildings = buildings
        self.building_to_weather = building_to_weather
        self.building_to_category = building_to_category
        self.categories = categories
        self.category_to_idx = {cat: i for i, cat in enumerate(categories)}
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.handle_missing = handle_missing
        self.time_range = time_range
        
        # 找出所有数据集中共有的时间戳
        common_timestamps = set(electricity_data.index)
        for building in buildings:
            weather_station = building_to_weather[building]
            if weather_station in weather_data and weather_data[weather_station] is not None:
                common_timestamps = common_timestamps.intersection(
                    set(weather_data[weather_station].index)
                )
        
        self.timestamps = sorted(list(common_timestamps))
        
        # 根据time_range限制时间戳范围
        if time_range != 'all':
            split_idx = int(len(self.timestamps) * 0.8)  # 80%/20%分割点
            if time_range == 'first_80_percent':
                self.timestamps = self.timestamps[:split_idx]
            elif time_range == 'last_20_percent':
                self.timestamps = self.timestamps[split_idx:]
        
        # 检测缺失
        self.has_missing = self.electricity_data.isna().any().any()
        if self.has_missing:
            missing_rate = self.electricity_data.isna().sum().sum() / \
                          (len(self.electricity_data) * len(self.electricity_data.columns)) * 100
            print(f"  ⚠️  数据包含缺失 (缺失率: {missing_rate:.2f}%)")
        
        print(f"  数据集: {len(buildings)} 个建筑, "
              f"{len(self.timestamps) - sequence_length - forecast_horizon} 个有效样本"
              f"{f' ({time_range})' if time_range != 'all' else ''}")
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return max(1, len(self.timestamps) - self.sequence_length - self.forecast_horizon)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str, torch.Tensor]:
        """
        获取一个样本
        
        返回:
            features: [buildings, sequence_length, features] 输入特征
            targets: [buildings, forecast_horizon] 预测目标
            category: 类别名称
            category_onehot: 类别的one-hot编码
        """
        # 计算时间窗口索引
        input_start_idx = idx
        input_end_idx = idx + self.sequence_length
        target_start_idx = input_end_idx
        target_end_idx = target_start_idx + self.forecast_horizon
        
        # 边界检查
        if target_end_idx > len(self.timestamps):
            target_end_idx = len(self.timestamps)
            target_start_idx = target_end_idx - self.forecast_horizon
            input_end_idx = target_start_idx
            input_start_idx = input_end_idx - self.sequence_length
        
        input_timestamps = self.timestamps[input_start_idx:input_end_idx]
        target_timestamps = self.timestamps[target_start_idx:target_end_idx]
        
        all_features = []
        all_targets = []
        
        # 为每个建筑提取数据
        for building in self.buildings:
            try:
                # 提取电力数据
                electricity_input = self.electricity_data.loc[input_timestamps, building].values
                electricity_target = self.electricity_data.loc[target_timestamps, building].values
                
                # 提取天气数据 - 修改这部分
                weather_station = self.building_to_weather[building]
                
                # 创建固定形状的天气数据数组 (sequence_length x 5)
                # 假设天气数据有5个标准特征：温度、露点、气压、风向、风速
                weather_features = 5
                weather_input = np.zeros((len(input_timestamps), weather_features))
                
                # 尝试填充实际天气数据
                if weather_station in self.weather_data and self.weather_data[weather_station] is not None:
                    # 获取天气数据的列名（排除site_id）
                    weather_columns = [col for col in self.weather_data[weather_station].columns 
                                      if col != 'site_id']
                    
                    # 确保我们只使用前5个特征（或更少，如果没有那么多）
                    weather_columns = weather_columns[:weather_features]
                    
                    # 填充天气数据
                    for i, ts in enumerate(input_timestamps):
                        if ts in self.weather_data[weather_station].index:
                            for j, col in enumerate(weather_columns):
                                if j < weather_features:  # 确保不超出预定义的特征数
                                    try:
                                        value = self.weather_data[weather_station].loc[ts, col]
                                        if isinstance(value, (int, float)):
                                            weather_input[i, j] = value
                                    except Exception as e:
                                        # 忽略错误，保持为0
                                        pass
                
                # 处理缺失值
                if self.handle_missing == 'forward_fill':
                    electricity_input = pd.Series(electricity_input).ffill().bfill().fillna(0).values
                    electricity_target = pd.Series(electricity_target).ffill().bfill().fillna(0).values
                    weather_input = pd.DataFrame(weather_input).ffill().bfill().fillna(0).values
                elif self.handle_missing == 'zero_fill':
                    electricity_input = np.nan_to_num(electricity_input, nan=0.0)
                    electricity_target = np.nan_to_num(electricity_target, nan=0.0)
                    weather_input = np.nan_to_num(weather_input, nan=0.0)
                
                # 确保形状正确
                electricity_input = electricity_input.reshape(-1, 1)
                
                # 长度检查和调整
                if len(electricity_input) != self.sequence_length:
                    electricity_input = np.resize(electricity_input, (self.sequence_length, 1))
                
                if len(weather_input) != self.sequence_length:
                    if len(weather_input) > self.sequence_length:
                        weather_input = weather_input[:self.sequence_length]
                    else:
                        padding = np.zeros((self.sequence_length - len(weather_input), weather_features))
                        weather_input = np.vstack([weather_input, padding])
                
                # 合并特征: [电力, 天气特征...]
                combined_features = np.concatenate([electricity_input, weather_input], axis=1)
                
                all_features.append(combined_features)
                all_targets.append(electricity_target)
                
            except Exception as e:
                print(f"⚠️  处理建筑 {building} 时出错: {str(e)}")
                # 使用零填充的默认数据
                combined_features = np.zeros((self.sequence_length, 1 + 5))  # 1个电力特征 + 5个天气特征
                electricity_target = np.zeros(self.forecast_horizon)
                all_features.append(combined_features)
                all_targets.append(electricity_target)
        
        # 转换为张量
        features_tensor = torch.tensor(np.array(all_features), dtype=torch.float32)
        targets_tensor = torch.tensor(np.array(all_targets), dtype=torch.float32)
        
        # 类别信息
        category = self.building_to_category[self.buildings[0]]
        category_idx = self.category_to_idx[category]
        category_onehot = torch.zeros(len(self.categories))
        category_onehot[category_idx] = 1
        
        return features_tensor, targets_tensor, category, category_onehot

def load_source_domain_dataloaders(
    batch_size: int = 32,
    sequence_length: int = 24,
    forecast_horizon: int = 24,
    handle_missing: str = 'forward_fill',
    val_ratio: float = 0.2
) -> Tuple[Optional[DataLoader], Optional[DataLoader], List[str]]:
    """
    加载源域训练数据（仅使用train_test_labels.json中标记为"train"的建筑）
    """
    print("="*70)
    print(f"📊 加载源域训练数据")
    print("="*70)
    print("说明:")
    print("  • 仅使用train_test_labels.json中标记为train的建筑")
    print("  • 使用完整数据（无人为引入的缺失）")
    print(f"  • 按时间划分: 前{(1-val_ratio)*100:.0f}%用于训练, 后{val_ratio*100:.0f}%用于验证")
    print("="*70)
    
    # 1. 加载配置文件
    with open('weather_labels.json', 'r') as f:
        weather_labels = json.load(f)
    
    with open('train_test_labels.json', 'r') as f:
        train_test_labels = json.load(f)
    
    # 2. 创建映射
    building_to_weather = {
        b["building_id"]: b["weather_station"]
        for b in weather_labels["buildings"]
    }
    
    building_to_category = {}
    categories = list(train_test_labels.keys())
    
    for cat in categories:
        train_buildings = train_test_labels[cat].get("train")
        if train_buildings:
            for building in train_buildings:
                building_to_category[building] = cat
        test_buildings = train_test_labels[cat].get("test", [])
        for building in test_buildings:
            building_to_category[building] = cat
    
    # 3. 收集所有训练建筑
    all_train_buildings = []
    
    for cat in categories:
        train_buildings = train_test_labels[cat].get("train")
        if train_buildings:
            all_train_buildings.extend(train_buildings)
    
    print(f"\n🏢 建筑分组:")
    print(f"  训练建筑: {len(all_train_buildings)} 个")
    
    # 4. 加载电力数据
    print(f"\n⚡ 加载电力数据...")
    
    full_electricity = pd.read_csv('data/electricity_train_buildings_only.csv')
    full_electricity['timestamp'] = pd.to_datetime(full_electricity['timestamp'])
    full_electricity.set_index('timestamp', inplace=True)
    print(f"  ✓ 完整电力数据: {full_electricity.shape}")
    
    # 5. 构建训练电力数据
    train_electricity = full_electricity[all_train_buildings].copy()
    print(f"  ✓ 训练集电力: {train_electricity.shape}")
    
    # 6. 加载天气数据
    print(f"\n🌤️  加载天气数据...")
    
    def load_weather_file(station: str) -> Optional[pd.DataFrame]:
        """加载天气数据文件"""
        path = f'data/weather_data/{station}_processed.csv'
        
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            if 'site_id' in df.columns:
                df = df.drop(columns=['site_id'])
            return df
        else:
            print(f"    ⚠️  找不到: {path}")
            return None  # ✅ 正确缩进
    
    # 获取天气站
    train_weather_stations = set([building_to_weather[b] for b in all_train_buildings])
    
    print("  训练集天气数据:")
    train_weather = {}
    for station in train_weather_stations:
        data = load_weather_file(station)
        train_weather[station] = data
        if data is not None:
            print(f"    ✓ {station}: {data.shape}")
    
    # 7. 创建训练和验证数据集
    print(f"\n🔄 创建数据加载器...")
    
    # 训练集 (前80%)
    train_dataset = ChronosDataset(
        electricity_data=train_electricity,
        weather_data=train_weather,
        buildings=all_train_buildings,
        building_to_weather=building_to_weather,
        building_to_category=building_to_category,
        categories=categories,
        sequence_length=sequence_length,
        forecast_horizon=forecast_horizon,
        handle_missing=handle_missing,
        time_range='first_80_percent'
    )
    
    # 验证集 (后20%)
    val_dataset = ChronosDataset(
        electricity_data=train_electricity,
        weather_data=train_weather,
        buildings=all_train_buildings,
        building_to_weather=building_to_weather,
        building_to_category=building_to_category,
        categories=categories,
        sequence_length=sequence_length,
        forecast_horizon=forecast_horizon,
        handle_missing=handle_missing,
        time_range='last_20_percent'
    )
    
    # ===== 修改为 =====
    # 创建数据加载器（使用自定义 collate 函数）
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=custom_collate_fn  # ✅ 添加此行
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=custom_collate_fn  # ✅ 添加此行
    )
    
    # 8. 总结
    print(f"\n" + "="*70)
    print(f"✅ 源域数据加载完成!")
    print("="*70)
    print(f"训练样本: {len(train_dataset)} (前80%时间)")
    print(f"验证样本: {len(val_dataset)} (后20%时间)")
    print(f"批次大小: {batch_size}")
    print(f"缺失处理: {handle_missing}")
    print("="*70 + "\n")
    
    return train_loader, val_loader, categories


def load_transfer_learning_dataloaders(
    category: str,
    data_shortage: str = 'mild',
    batch_size: int = 32,
    sequence_length: int = 24,
    forecast_horizon: int = 24,
    handle_missing: str = 'forward_fill'
) -> Tuple[Optional[DataLoader], Optional[DataLoader], List[str]]:
    """
    加载迁移学习数据加载器
    
    训练数据:
    - 使用train_test_labels.json中标记为"train"的建筑数据（完整数据）
    - 加上稀缺度划分的训练集部分（前80%带缺失的数据）
    
    测试数据:
    - 使用稀缺度划分的测试集部分（后20%完整数据）
    
    特殊情况 - CC类别:
    - 如果类别是CC（没有训练建筑），从其他5个类别各随机选1个建筑
    """
    print("="*70)
    print(f"📊 加载迁移学习数据 - 类别: {category}, 稀缺度: {data_shortage.upper()}")
    print("="*70)
    print("说明:")
    print("  • 训练数据: train标签建筑(完整) + 目标建筑稀缺训练集(前80%)")
    print("  • 测试数据: 目标建筑稀缺测试集(后20%)")
    if category == 'CC':
        print("  • 特殊处理: CC类别无训练建筑，从其他5类各随机选1个建筑")
    print("="*70)
    
    # 1. 加载配置文件
    with open('weather_labels.json', 'r') as f:
        weather_labels = json.load(f)
    
    with open('train_test_labels.json', 'r') as f:
        train_test_labels = json.load(f)
    
    # 2. 创建映射
    building_to_weather = {
        b["building_id"]: b["weather_station"]
        for b in weather_labels["buildings"]
    }
    
    building_to_category = {}
    categories = list(train_test_labels.keys())
    
    for cat in categories:
        train_buildings = train_test_labels[cat].get("train")
        if train_buildings:
            for building in train_buildings:
                building_to_category[building] = cat
        test_buildings = train_test_labels[cat].get("test", [])
        for building in test_buildings:
            building_to_category[building] = cat
    
    # 3. 获取训练和测试建筑
    if category == 'CC':
        # CC类别特殊处理：从其他5个类别各随机选1个建筑
        train_buildings = []
        for cat in categories:
            if cat != 'CC' and train_test_labels[cat].get("train"):
                selected_building = random.choice(train_test_labels[cat]["train"])
                train_buildings.append(selected_building)
                print(f"  从类别 {cat} 随机选择建筑: {selected_building}")
    else:
        # 正常类别：使用该类别的train标签建筑
        train_buildings = train_test_labels[category].get("train", [])
        if train_buildings is None:
            train_buildings = []
    
    # 获取目标建筑（测试建筑）
    test_buildings = train_test_labels[category].get("test", [])
    
    if not test_buildings:
        print(f"❌ 错误: 类别 {category} 没有测试建筑")
        return None, None, categories
    
    print(f"\n🏢 建筑分组:")
    print(f"  训练建筑: {len(train_buildings)} 个 - {train_buildings}")
    print(f"  目标建筑: {len(test_buildings)} 个 - {test_buildings}")
    
    # 4. 加载电力数据
    print(f"\n⚡ 加载电力数据...")
    
    # 完整电力数据（用于训练建筑）
    full_electricity = pd.read_csv('data/electricity_train_buildings_only.csv')
    full_electricity['timestamp'] = pd.to_datetime(full_electricity['timestamp'])
    full_electricity.set_index('timestamp', inplace=True)
    
    # 稀缺电力数据（用于目标建筑）
    scarcity_electricity = pd.read_csv(
        f'data/target_scarcity_data/scarcity_data/data_{data_shortage}_scarcity.csv'
    )
    scarcity_electricity['timestamp'] = pd.to_datetime(scarcity_electricity['timestamp'])
    scarcity_electricity.set_index('timestamp', inplace=True)
    
    # 稀缺训练集和测试集（按时间划分）
    split_idx = int(len(scarcity_electricity) * 0.8)
    scarcity_timestamps = scarcity_electricity.index.tolist()
    train_timestamps = scarcity_timestamps[:split_idx]
    test_timestamps = scarcity_timestamps[split_idx:]
    
    # 训练集电力数据
    train_scarcity_electricity = scarcity_electricity.loc[train_timestamps]
    
    # 测试集电力数据
    test_scarcity_electricity = scarcity_electricity.loc[test_timestamps]
    
    print(f"  ✓ 完整电力数据: {full_electricity.shape}")
    print(f"  ✓ 稀缺电力数据: {scarcity_electricity.shape}")
    print(f"  ✓ 稀缺训练集: {train_scarcity_electricity.shape} (前80%)")
    print(f"  ✓ 稀缺测试集: {test_scarcity_electricity.shape} (后20%)")
    
    # 5. 加载天气数据
    print(f"\n🌤️  加载天气数据...")
    
    def load_weather_file(station: str, is_complete: bool = True) -> Optional[pd.DataFrame]:
        """加载天气数据文件"""
        if is_complete:
            path = f'data/weather_data/{station}_processed.csv'
        else:
            path = f'data/target_scarcity_data/scarcity_data/weather/{station}_{data_shortage}_scarcity.csv'
        
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            if 'site_id' in df.columns:
                df = df.drop(columns=['site_id'])
            return df
        else:
            print(f"    ⚠️  找不到: {path}")
            return None
    
    # 获取天气站
    train_weather_stations = set([building_to_weather[b] for b in train_buildings]) if train_buildings else set()
    test_weather_stations = set([building_to_weather[b] for b in test_buildings])
    
    # 训练建筑使用完整天气数据
    train_weather = {}
    if train_buildings:
        print("  训练建筑天气数据 (完整):")
        for station in train_weather_stations:
            data = load_weather_file(station, is_complete=True)
            train_weather[station] = data
            if data is not None:
                print(f"    ✓ {station}: {data.shape}")
    
    # 目标建筑使用稀缺天气数据
    print(f"  目标建筑天气数据 ({data_shortage}):")
    test_weather = {}
    for station in test_weather_stations:
        data = load_weather_file(station, is_complete=False)
        test_weather[station] = data
        if data is not None:
            missing = data.isna().sum().sum() / (len(data) * len(data.columns)) * 100
            print(f"    ✓ {station}: {data.shape} (缺失率: {missing:.2f}%)")
    
    # 6. 创建训练和测试数据集
    print(f"\n🔄 创建数据加载器...")
    
    # 训练建筑数据集
    train_buildings_dataset = None
    if train_buildings:
        print(f"  创建训练建筑数据集 ({len(train_buildings)} 个建筑)...")
        train_buildings_dataset = ChronosDataset(
            electricity_data=full_electricity[train_buildings],
            weather_data=train_weather,
            buildings=train_buildings,
            building_to_weather=building_to_weather,
            building_to_category=building_to_category,
            categories=categories,
            sequence_length=sequence_length,
            forecast_horizon=forecast_horizon,
            handle_missing=handle_missing,
            time_range='all'
        )
    
    # 目标建筑训练集数据集
    print(f"  创建目标建筑训练集数据集 ({len(test_buildings)} 个建筑)...")
    target_train_dataset = ChronosDataset(
        electricity_data=train_scarcity_electricity[test_buildings],
        weather_data=test_weather,
        buildings=test_buildings,
        building_to_weather=building_to_weather,
        building_to_category=building_to_category,
        categories=categories,
        sequence_length=sequence_length,
        forecast_horizon=forecast_horizon,
        handle_missing=handle_missing,
        time_range='all'
    )
    
    # 目标建筑测试集数据集
    print(f"  创建目标建筑测试集数据集 ({len(test_buildings)} 个建筑)...")
    target_test_dataset = ChronosDataset(
        electricity_data=test_scarcity_electricity[test_buildings],
        weather_data=test_weather,
        buildings=test_buildings,
        building_to_weather=building_to_weather,
        building_to_category=building_to_category,
        categories=categories,
        sequence_length=sequence_length,
        forecast_horizon=forecast_horizon,
        handle_missing=handle_missing,
        time_range='all'
    )
    
    # 合并训练数据集
    if train_buildings_dataset is not None:
        combined_train_dataset = ConcatDataset([train_buildings_dataset, target_train_dataset])
    else:
        combined_train_dataset = target_train_dataset
    
    # 创建数据加载器
    # 创建数据加载器（使用自定义 collate 函数）
    train_loader = DataLoader(
        combined_train_dataset,
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0, 
        pin_memory=True,
        collate_fn=custom_collate_fn  # ✅ 添加此行
    )
    
    test_loader = DataLoader(
        target_test_dataset,
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0, 
        pin_memory=True,
        collate_fn=custom_collate_fn  # ✅ 添加此行
    )
    
    # 7. 总结
    print(f"\n" + "="*70)
    print(f"✅ 类别 {category} 迁移学习数据加载完成!")
    print("="*70)
    print(f"训练样本: {len(combined_train_dataset)}")
    if train_buildings_dataset is not None:
        print(f"  - 训练建筑数据: {len(train_buildings_dataset)} 样本")
    print(f"  - 目标建筑训练数据: {len(target_train_dataset)} 样本")
    print(f"测试样本: {len(target_test_dataset)}")
    print(f"批次大小: {batch_size}")
    print(f"缺失处理: {handle_missing}")
    print("="*70 + "\n")
    
    return train_loader, test_loader, categories
    
def load_pure_target_dataloaders(
    category: str,
    data_shortage: str = 'mild',
    batch_size: int = 32,
    sequence_length: int = 24,
    forecast_horizon: int = 24,
    handle_missing: str = 'forward_fill'
) -> Tuple[Optional[DataLoader], Optional[DataLoader], List[str]]:
    """
    [新增函数] 加载纯目标域数据加载器 (用于 Pure BiLSTM 对比实验)
    
    特点:
    - 仅加载目标域的稀缺数据
    - 严禁包含源域（Source Domain）的完整数据
    - 用于验证在无迁移学习情况下，仅靠稀缺数据训练的效果
    """
    print("="*70)
    print(f"📉 加载纯目标域数据 (无迁移) - 类别: {category}, 稀缺度: {data_shortage.upper()}")
    print("="*70)
    
    # 1. 加载配置文件
    with open('weather_labels.json', 'r') as f:
        weather_labels = json.load(f)
    
    with open('train_test_labels.json', 'r') as f:
        train_test_labels = json.load(f)
    
    # 2. 创建映射
    building_to_weather = {
        b["building_id"]: b["weather_station"]
        for b in weather_labels["buildings"]
    }
    
    building_to_category = {}
    categories = list(train_test_labels.keys())
    
    # 3. 获取目标建筑 (测试建筑)
    test_buildings = train_test_labels[category].get("test", [])
    
    # 更新映射
    for building in test_buildings:
        building_to_category[building] = category
    
    if not test_buildings:
        print(f"❌ 错误: 类别 {category} 没有测试建筑")
        return None, None, categories
        
    print(f"  目标建筑 (仅使用这些): {len(test_buildings)} 个 - {test_buildings}")
    
    # 4. 加载稀缺电力数据
    print(f"\n⚡ 加载稀缺电力数据...")
    scarcity_path = f'data/target_scarcity_data/scarcity_data/data_{data_shortage}_scarcity.csv'
    
    if not os.path.exists(scarcity_path):
        print(f"❌ 找不到文件: {scarcity_path}")
        return None, None, categories

    scarcity_electricity = pd.read_csv(scarcity_path)
    scarcity_electricity['timestamp'] = pd.to_datetime(scarcity_electricity['timestamp'])
    scarcity_electricity.set_index('timestamp', inplace=True)
    
    # 划分训练集和测试集 (按时间划分 80/20)
    split_idx = int(len(scarcity_electricity) * 0.8)
    timestamps = scarcity_electricity.index.tolist()
    train_timestamps = timestamps[:split_idx]
    test_timestamps = timestamps[split_idx:]
    
    # 5. 加载稀缺天气数据
    print(f"\n🌤️  加载稀缺天气数据 ({data_shortage})...")
    test_weather_stations = set([building_to_weather[b] for b in test_buildings])
    
    weather_data = {}
    for station in test_weather_stations:
        path = f'data/target_scarcity_data/scarcity_data/weather/{station}_{data_shortage}_scarcity.csv'
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            if 'site_id' in df.columns:
                df = df.drop(columns=['site_id'])
            weather_data[station] = df
        else:
            print(f"    ⚠️  找不到天气数据: {path}")

    # 6. 创建数据集
    print(f"\n🔄 创建纯目标域数据集...")
    
    # 目标域训练集 (使用稀缺数据的前80%)
    target_train_dataset = ChronosDataset(
        electricity_data=scarcity_electricity.loc[train_timestamps, test_buildings],
        weather_data=weather_data,
        buildings=test_buildings,
        building_to_weather=building_to_weather,
        building_to_category=building_to_category,
        categories=categories,
        sequence_length=sequence_length,
        forecast_horizon=forecast_horizon,
        handle_missing=handle_missing,
        time_range='all' # 数据已经通过loc筛选过了，这里用all即可
    )
    
    # 目标域测试集 (使用稀缺数据的后20% - 实际上这部分是完整的，因为测试集通常假设有GT)
    # 注意：根据你的逻辑，target_scarcity_data 里的后20%应该是用于评估的
    target_test_dataset = ChronosDataset(
        electricity_data=scarcity_electricity.loc[test_timestamps, test_buildings],
        weather_data=weather_data,
        buildings=test_buildings,
        building_to_weather=building_to_weather,
        building_to_category=building_to_category,
        categories=categories,
        sequence_length=sequence_length,
        forecast_horizon=forecast_horizon,
        handle_missing=handle_missing,
        time_range='all'
    )
    
    # 7. 创建 DataLoader
    # 关键点：这里只使用 target_train_dataset，没有 ConcatDataset
    train_loader = DataLoader(
        target_train_dataset,
        batch_size=batch_size, 
        shuffle=True, # 训练时打乱
        num_workers=0, 
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    
    test_loader = DataLoader(
        target_test_dataset,
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0, 
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    
    print(f"✅ 纯目标域数据加载完成: 训练集 {len(target_train_dataset)} 样本, 测试集 {len(target_test_dataset)} 样本")
    return train_loader, test_loader, categories