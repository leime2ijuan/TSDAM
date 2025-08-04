import pandas as pd
import numpy as np
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader

class ChronosDataset(Dataset):
    def __init__(self, electricity_data, weather_data, buildings, building_to_weather, 
             building_to_category, categories, sequence_length=24, forecast_horizon=24):
        self.electricity_data = electricity_data
        self.weather_data = weather_data
        self.buildings = buildings
        self.building_to_weather = building_to_weather
        self.building_to_category = building_to_category
        self.categories = categories
        self.category_to_idx = {cat: i for i, cat in enumerate(categories)}
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        

        
        # 找出所有数据集中共有的时间戳
        common_timestamps = set(electricity_data.index)
        for building in buildings:
            weather_station = building_to_weather[building]
            if weather_station in weather_data and weather_data[weather_station] is not None:
                common_timestamps = common_timestamps.intersection(set(weather_data[weather_station].index))
        
        # 转换为排序后的列表
        self.timestamps = sorted(list(common_timestamps))
        
        print(f"Chronos数据集初始化: {len(buildings)} 个建筑, {len(self.timestamps)-sequence_length-forecast_horizon} 个有效样本")
    def __len__(self):
        # 确保数据集长度至少为1
        length = max(1, len(self.timestamps) - self.sequence_length - self.forecast_horizon)
        return length        
    def _preprocess_weather_data(self):
        """预处理天气数据，确保所有时间戳和数据长度一致"""
        for station, data in self.weather_data.items():
            if data is None:
                continue
                
            # 检查是否有特定建筑的天气数据长度异常
            for building in self.buildings:
                if self.building_to_weather[building] == station:
                    print(f"天气站 {station} 对应建筑 {building} 的数据长度: {len(data)}")
    
    def __getitem__(self, idx):
        # 获取输入序列和目标序列的时间窗口
        input_start_idx = idx
        input_end_idx = idx + self.sequence_length
        target_start_idx = input_end_idx
        target_end_idx = target_start_idx + self.forecast_horizon
        
        if target_end_idx > len(self.timestamps):
            target_end_idx = len(self.timestamps)
            target_start_idx = target_end_idx - self.forecast_horizon
            input_end_idx = target_start_idx
            input_start_idx = input_end_idx - self.sequence_length
        
        input_timestamps = self.timestamps[input_start_idx:input_end_idx]
        target_timestamps = self.timestamps[target_start_idx:target_end_idx]
        
        # 验证时间窗口长度
        if len(input_timestamps) != self.sequence_length:
            print(f"警告: 输入时间窗口长度为 {len(input_timestamps)}，而不是预期的 {self.sequence_length}")
        if len(target_timestamps) != self.forecast_horizon:
            print(f"警告: 目标时间窗口长度为 {len(target_timestamps)}，而不是预期的 {self.forecast_horizon}")
        
        # 准备所有建筑物的特征和目标
        all_features = []
        all_targets = []
        
        for building in self.buildings:
            try:
                # 确保使用准确的时间戳索引
                if not all(ts in self.electricity_data.index for ts in input_timestamps):
                    missing = [ts for ts in input_timestamps if ts not in self.electricity_data.index]
                    print(f"警告: 建筑 {building} 的输入时间戳中有 {len(missing)} 个不在电力数据中")
                
                electricity_input = self.electricity_data.loc[input_timestamps, building].values
                electricity_target = self.electricity_data.loc[target_timestamps, building].values
                
                # 获取对应的天气数据
                weather_station = self.building_to_weather[building]
                
                # 确保所有时间戳都在天气数据中
                if not all(ts in self.weather_data[weather_station].index for ts in input_timestamps):
                    missing = [ts for ts in input_timestamps if ts not in self.weather_data[weather_station].index]
                    print(f"警告: 建筑 {building} 的输入时间戳中有 {len(missing)} 个不在天气数据中")
                
                # 严格按照输入时间戳获取天气数据
                weather_rows = []
                for ts in input_timestamps:
                    if ts in self.weather_data[weather_station].index:
                        weather_rows.append(self.weather_data[weather_station].loc[ts].values)
                    else:
                        # 使用零填充缺失数据
                        print(f"警告: 时间戳 {ts} 在天气站 {weather_station} 数据中缺失")
                        weather_rows.append(np.zeros_like(self.weather_data[weather_station].iloc[0].values))
                
                weather_input = np.array(weather_rows)
                
                # 确保数据长度匹配
                if len(electricity_input) != self.sequence_length:
                    print(f"警告: 建筑 {building} 的电力输入数据长度为 {len(electricity_input)}，而不是预期的 {self.sequence_length}")
                    if len(electricity_input) > self.sequence_length:
                        electricity_input = electricity_input[:self.sequence_length]
                    else:
                        electricity_input = np.resize(electricity_input, (self.sequence_length,))
                
                # 处理天气数据
                if weather_input.ndim == 1:
                    weather_input = weather_input.reshape(-1, 1)
                
                # 严格检查天气数据长度
                if len(weather_input) != self.sequence_length:
                    print(f"警告: 建筑 {building} 的天气输入数据长度为 {len(weather_input)}，而不是预期的 {self.sequence_length}")
                    if len(weather_input) > self.sequence_length:
                        # 使用切片而不是resize
                        weather_input = weather_input[:self.sequence_length]
                    else:
                        # 如果长度不足，复制最后一行填充
                        padding = np.tile(weather_input[-1:], (self.sequence_length - len(weather_input), 1))
                        weather_input = np.vstack([weather_input, padding])
                
                # 将电力数据添加为第一个特征
                electricity_input = electricity_input.reshape(-1, 1)
                
                # 最终检查并确保两个数组在第一维上长度相同
                assert electricity_input.shape[0] == weather_input.shape[0], f"电力数据形状 {electricity_input.shape} 与天气数据形状 {weather_input.shape} 不匹配"
                
                # 合并电力和天气特征
                combined_features = np.concatenate([electricity_input, weather_input], axis=1)
                
                all_features.append(combined_features)
                all_targets.append(electricity_target)
                
            except Exception as e:
                print(f"处理建筑 {building} 数据时出错: {str(e)}")
                # 创建默认数据
                num_weather_features = next(iter(self.weather_data.values())).shape[1] if self.weather_data else 5
                combined_features = np.zeros((self.sequence_length, 1 + num_weather_features))
                electricity_target = np.zeros(self.forecast_horizon)
                
                all_features.append(combined_features)
                all_targets.append(electricity_target)
        
        # 额外验证：确保所有特征具有相同的形状
        feature_shapes = [f.shape for f in all_features]
        if len(set(str(s) for s in feature_shapes)) > 1:
            print(f"警告: 不同建筑的特征形状不一致: {feature_shapes}")
            # 找到最常见的形状
            from collections import Counter
            shape_counter = Counter([str(s) for s in feature_shapes])
            most_common_shape_str = shape_counter.most_common(1)[0][0]
            # 将字符串形状转换回元组
            import ast
            most_common_shape = ast.literal_eval(most_common_shape_str.replace('(', '').replace(')', ''))
            
            # 调整所有特征到相同形状
            for i in range(len(all_features)):
                if all_features[i].shape != most_common_shape:
                    print(f"调整第 {i} 个建筑的特征形状从 {all_features[i].shape} 到 {most_common_shape}")
                    temp = np.zeros(most_common_shape)
                    # 复制尽可能多的数据
                    min_rows = min(all_features[i].shape[0], most_common_shape[0])
                    min_cols = min(all_features[i].shape[1], most_common_shape[1])
                    temp[:min_rows, :min_cols] = all_features[i][:min_rows, :min_cols]
                    all_features[i] = temp
        
        # 转换为张量
        features_tensor = torch.tensor(np.array(all_features), dtype=torch.float32)
        targets_tensor = torch.tensor(np.array(all_targets), dtype=torch.float32)
        
        # 获取类别信息
        category = self.building_to_category[self.buildings[0]]  # 假设所有建筑属于同一类别
        category_idx = self.category_to_idx[category]
        category_onehot = torch.zeros(len(self.categories))
        category_onehot[category_idx] = 1
        
        return features_tensor, targets_tensor, category, category_onehot
def create_all_dataloaders(batch_size=32, sequence_length=24, forecast_horizon=24):
    # 加载配置文件
    with open('weather_labels.json', 'r') as f:
        weather_labels = json.load(f)

    with open('train_test_labels.json', 'r') as f:
        train_test_labels = json.load(f)

    # 创建建筑ID到天气站的映射
    building_to_weather = {}
    for building in weather_labels["buildings"]:
        building_to_weather[building["building_id"]] = building["weather_station"]

    # 创建建筑ID到类别的映射
    building_to_category = {}
    categories = list(train_test_labels.keys())
    for category in categories:
        # 处理训练建筑
        train_buildings = train_test_labels[category]["train"]
        if train_buildings is not None:  # 新增：判断train是否为null
            for building in train_buildings:
                building_to_category[building] = category
        
        # 处理测试建筑
        test_buildings = train_test_labels[category]["test"]
        for building in test_buildings:
            building_to_category[building] = category

    # 加载电力数据
    train_electricity = pd.read_csv('electricity_train_data/electricity_mild_train.csv')
    train_electricity['timestamp'] = pd.to_datetime(train_electricity['timestamp'])
    train_electricity.set_index('timestamp', inplace=True)

    test_electricity = pd.read_csv('electricity_train_data/electricity_mild_test.csv')
    test_electricity['timestamp'] = pd.to_datetime(test_electricity['timestamp'])
    test_electricity.set_index('timestamp', inplace=True)

    def load_weather_data(station, data_type='mild', train_or_test='train'):
        file_path = f'weather_train_data/{station}_{data_type}_{train_or_test}.csv'
        if os.path.exists(file_path):
            data = pd.read_csv(file_path)
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            
            # 直接设置时间戳为索引，不检查重复
            data.set_index('timestamp', inplace=True)
            
            # 移除非数值列
            if 'site_id' in data.columns:
                data = data.drop(columns=['site_id'])
                
            print(f"加载天气数据: {station}_{data_type}_{train_or_test}.csv, 形状: {data.shape}")
            
            return data
        else:
            print(f"警告: 找不到天气数据文件 {file_path}")
            return None

    # 加载所有天气站的数据
    weather_stations = set(building_to_weather.values())
    train_weather = {}
    test_weather = {}

    for station in weather_stations:
        train_weather[station] = load_weather_data(station, 'mild', 'train')
        test_weather[station] = load_weather_data(station, 'mild', 'test')

    # 按类别组织训练和测试建筑
    train_buildings_by_category = {}
    test_buildings_by_category = {}
    
    for category in categories:
        # 新增：处理train为null的情况
        train_buildings = train_test_labels[category]["train"]
        if train_buildings is None:
            train_buildings_by_category[category] = []  # 设置为空列表
        else:
            train_buildings_by_category[category] = train_buildings
        
        test_buildings_by_category[category] = train_test_labels[category]["test"]
    
    # 创建每个类别的数据集
    train_datasets = []
    test_datasets = []
    
    for category in categories:
        # 训练数据集：仅当train_buildings不为null且非空时创建
        train_buildings = train_buildings_by_category[category]
        if train_buildings is not None and len(train_buildings) > 0:
            train_dataset = ChronosDataset(
                electricity_data=train_electricity,
                weather_data=train_weather,
                buildings=train_buildings,
                building_to_weather=building_to_weather,
                building_to_category=building_to_category,
                categories=categories,
                sequence_length=sequence_length,
                forecast_horizon=forecast_horizon
            )
            # 检查数据集长度
            dataset_length = len(train_dataset)
            print(f"类别 {category} 训练数据集长度: {dataset_length}")
            if dataset_length > 0:
                train_datasets.append(train_dataset)
            else:
                print(f"警告: 类别 {category} 的训练数据集长度为0，跳过")
        else:
            print(f"跳过类别 {category} 的训练数据集创建 (train为null或空)")
        
        # 测试数据集：无论train是否为null，正常处理测试集
        test_buildings = test_buildings_by_category[category]
        if test_buildings:
            test_dataset = ChronosDataset(
                electricity_data=test_electricity,
                weather_data=test_weather,
                buildings=test_buildings,
                building_to_weather=building_to_weather,
                building_to_category=building_to_category,
                categories=categories,
                sequence_length=sequence_length,
                forecast_horizon=forecast_horizon
            )
            # 检查数据集长度
            dataset_length = len(test_dataset)
            print(f"类别 {category} 测试数据集长度: {dataset_length}")
            if dataset_length > 0:
                test_datasets.append(test_dataset)
            else:
                print(f"警告: 类别 {category} 的测试数据集长度为0，跳过")
        else:
            print(f"警告: 类别 {category} 的测试建筑列表为空，跳过")
    
    # 合并所有类别的数据集
    from torch.utils.data import ConcatDataset
    
    if train_datasets:
        train_dataset = ConcatDataset(train_datasets)
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
    else:
        train_dataloader = None
    
    if test_datasets:
        test_dataset = ConcatDataset(test_datasets)
        test_dataloader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
    else:
        test_dataloader = None
    
    return train_dataloader, test_dataloader, categories

def create_category_dataloaders(category, batch_size=32, sequence_length=24, forecast_horizon=24, data_shortage='mild'):
    """
    加载目标类别在指定数据短缺场景下的训练和测试数据加载器。
    
    参数:
    - category: 类别（如 "DO", "HO"）
    - batch_size: 批次大小
    - sequence_length: 输入序列长度
    - forecast_horizon: 预测步长
    - data_shortage: 'mild', 'heavy', 'extreme'
    """
    import os
    import json
    import pandas as pd
    from torch.utils.data import DataLoader

    with open('weather_labels.json', 'r') as f:
        weather_labels = json.load(f)

    with open('train_test_labels.json', 'r') as f:
        train_test_labels = json.load(f)

    building_to_weather = {
        b["building_id"]: b["weather_station"]
        for b in weather_labels["buildings"]
    }

    building_to_category = {}
    categories = list(train_test_labels.keys())
    for cat in categories:
        # 处理训练建筑（增加对train为null的检查）
        train_buildings = train_test_labels[cat]["train"]
        if train_buildings is not None:
            for building in train_buildings:
                building_to_category[building] = cat
        # 处理测试建筑
        for building in train_test_labels[cat]["test"]:
            building_to_category[building] = cat

    # 加载电力数据（根据 data_shortage）
    train_electricity = pd.read_csv(f'electricity_train_data/electricity_{data_shortage}_train.csv')
    train_electricity['timestamp'] = pd.to_datetime(train_electricity['timestamp'])
    train_electricity.set_index('timestamp', inplace=True)

    test_electricity = pd.read_csv(f'electricity_train_data/electricity_{data_shortage}_test.csv')
    test_electricity['timestamp'] = pd.to_datetime(test_electricity['timestamp'])
    test_electricity.set_index('timestamp', inplace=True)

    def load_weather_data(station, split):
        file_path = f'weather_train_data/{station}_{data_shortage}_{split}.csv'
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.drop_duplicates(subset=['timestamp'], keep='first')
            df.set_index('timestamp', inplace=True)
            if 'site_id' in df.columns:
                df = df.drop(columns=['site_id'])
            return df
        else:
            print(f"警告: {file_path} 不存在")
            return None

    # 获取当前类别的建筑
    train_buildings = train_test_labels[category]["train"]
    test_buildings = train_test_labels[category]["test"]

    # 新增：处理train为null的情况
    if train_buildings is None:
        train_buildings = []
        print(f"类别 {category} 的训练数据为空 (train=null)")

    # 加载天气数据（只加载涉及的站点）
    # 注意：即使train为空，仍需加载测试建筑对应的天气数据
    weather_stations = set([building_to_weather[b] for b in train_buildings + test_buildings])
    train_weather = {s: load_weather_data(s, 'train') for s in weather_stations}
    test_weather = {s: load_weather_data(s, 'test') for s in weather_stations}

    # 构建 Chronos 数据集（修改：根据train_buildings是否为空决定是否创建训练集）
    if train_buildings:
        train_dataset = ChronosDataset(
            electricity_data=train_electricity,
            weather_data=train_weather,
            buildings=train_buildings,
            building_to_weather=building_to_weather,
            building_to_category=building_to_category,
            categories=categories,
            sequence_length=sequence_length,
            forecast_horizon=forecast_horizon
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    else:
        train_loader = None
        print(f"类别 {category} 没有训练数据，train_loader设为None")

    # 测试数据集（保持不变）
    test_dataset = ChronosDataset(
        electricity_data=test_electricity,
        weather_data=test_weather,
        buildings=test_buildings,
        building_to_weather=building_to_weather,
        building_to_category=building_to_category,
        categories=categories,
        sequence_length=sequence_length,
        forecast_horizon=forecast_horizon
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    return train_loader, test_loader, categories


def create_transfer_dataloaders(category, data_shortage='mild', 
                              batch_size=32, sequence_length=24, forecast_horizon=24):
    """
    创建用于迁移学习的源域和目标域数据加载器
    
    参数:
    - category: 要处理的类别（如"DO", "HO"等）
    - data_shortage: 'mild', 'heavy', 或 'extreme'，表示数据短缺程度
    - batch_size: 批次大小
    - sequence_length: 输入序列长度
    - forecast_horizon: 预测时间范围
    
    返回:
    - source_dataloader: 源域数据加载器
    - target_dataloader: 目标域数据加载器
    - categories: 类别列表
    """
    # 加载配置文件
    with open('weather_labels.json', 'r') as f:
        weather_labels = json.load(f)

    with open('train_test_labels.json', 'r') as f:
        train_test_labels = json.load(f)

    # 创建建筑ID到天气站的映射
    building_to_weather = {}
    for building in weather_labels["buildings"]:
        building_to_weather[building["building_id"]] = building["weather_station"]

    # 创建建筑ID到类别的映射（增加对train为null或空的判断）
    building_to_category = {}
    categories = list(train_test_labels.keys())
    for cat in categories:
        train_buildings = train_test_labels[cat]["train"]
        if train_buildings is not None and len(train_buildings) > 0:
            for building in train_buildings:
                building_to_category[building] = cat
        test_buildings = train_test_labels[cat]["test"]
        if test_buildings is not None and len(test_buildings) > 0:
            for building in test_buildings:
                building_to_category[building] = cat

    # 获取当前类别的源建筑（训练集）和目标建筑（测试集）
    source_buildings = train_test_labels[category]["train"]
    test_buildings = train_test_labels[category]["test"]
    
    # 处理train为null或空的情况
    if source_buildings is None:
        source_buildings = []
        print(f"类别 {category} 的训练数据为空 (train=null)")
    
    # 确保测试建筑存在
    if not test_buildings or len(test_buildings) == 0:
        print(f"警告: 类别 {category} 没有测试建筑，无法创建目标域数据加载器")
        return None, None, categories
    
    target_building = test_buildings[0]  # 假设每个类别至少有一个测试建筑
    source_domain_buildings = source_buildings + [target_building]
    
    # 加载电力数据 - 训练和测试数据
    print(f"加载 {data_shortage} 场景的电力数据...")
    try:
        train_electricity = pd.read_csv(f'electricity_train_data/electricity_{data_shortage}_train.csv')
        train_electricity['timestamp'] = pd.to_datetime(train_electricity['timestamp'])
        train_electricity.set_index('timestamp', inplace=True)
        
        test_electricity = pd.read_csv(f'electricity_train_data/electricity_{data_shortage}_test.csv')
        test_electricity['timestamp'] = pd.to_datetime(test_electricity['timestamp'])
        test_electricity.set_index('timestamp', inplace=True)
    except Exception as e:
        print(f"加载电力数据时出错: {e}")
        return None, None, categories
    
    # 准备源域数据：源建筑的全年数据 + 目标建筑的训练数据
    print(f"准备源域数据...源建筑: {source_buildings} 和目标建筑(训练部分): {target_building}")
    try:
        # 合并训练和测试电力数据获取全年数据
        all_electricity = pd.concat([train_electricity, test_electricity], ignore_index=True)
        all_electricity = all_electricity.sort_values('timestamp')
        all_electricity.set_index('timestamp', inplace=True)
        
        # 提取源建筑的电力数据
        if source_buildings:  # 仅当源建筑存在时处理
            source_electricity = all_electricity[source_buildings].copy()
            source_electricity[target_building] = train_electricity[target_building]
        else:
            # 如果源建筑为空，使用目标建筑的训练数据作为源数据
            source_electricity = pd.DataFrame()
            source_electricity[target_building] = train_electricity[target_building]
            print(f"类别 {category} 没有源建筑数据，使用目标建筑的训练数据作为源域数据")
    except Exception as e:
        print(f"准备源域电力数据时出错: {e}")
        return None, None, categories
    
    # 准备目标域数据：目标建筑的测试数据
    print(f"准备目标域数据...目标建筑: {target_building}")
    try:
        target_electricity = test_electricity[[target_building]].copy()
    except Exception as e:
        print(f"准备目标域电力数据时出错: {e}")
        return None, None, categories
    
    # 加载天气数据
    print("加载天气数据...")
    weather_stations = set([building_to_weather.get(building) for building in source_domain_buildings])
    weather_stations = {s for s in weather_stations if s is not None}  # 过滤None值
    
    train_weather = {}
    test_weather = {}
    all_weather = {}
    
    for station in weather_stations:
        train_file = f'weather_train_data/{station}_{data_shortage}_train.csv'
        test_file = f'weather_train_data/{station}_{data_shortage}_test.csv'
        
        train_data = None
        test_data = None
        
        if os.path.exists(train_file):
            train_data = pd.read_csv(train_file)
            train_data['timestamp'] = pd.to_datetime(train_data['timestamp'])
            train_data.set_index('timestamp', inplace=True)
            
            # 移除非数值列
            if 'site_id' in train_data.columns:
                train_data = train_data.drop(columns=['site_id'])
                
            train_weather[station] = train_data
        else:
            print(f"警告: 找不到天气数据文件 {train_file}")
            train_weather[station] = None
            
        if os.path.exists(test_file):
            test_data = pd.read_csv(test_file)
            test_data['timestamp'] = pd.to_datetime(test_data['timestamp'])
            test_data.set_index('timestamp', inplace=True)
            
            # 移除非数值列
            if 'site_id' in test_data.columns:
                test_data = test_data.drop(columns=['site_id'])
                
            test_weather[station] = test_data
        else:
            print(f"警告: 找不到天气数据文件 {test_file}")
            test_weather[station] = None
        
        # 合并训练和测试天气数据，获取全年天气数据
        if train_data is not None and test_data is not None:
            all_data = pd.concat([train_data, test_data])
            all_data = all_data.sort_index()
            all_weather[station] = all_data
        elif train_data is not None:
            all_weather[station] = train_data
        elif test_data is not None:
            all_weather[station] = test_data
        else:
            all_weather[station] = None
    
    # 创建源域数据集
    print(f"创建源域数据集...源建筑: {source_domain_buildings}")
    try:
        source_dataset = ChronosDataset(
            electricity_data=source_electricity,
            weather_data=all_weather,
            buildings=source_domain_buildings,
            building_to_weather=building_to_weather,
            building_to_category=building_to_category,
            categories=categories,
            sequence_length=sequence_length,
            forecast_horizon=forecast_horizon
        )
        
        source_dataloader = DataLoader(
            source_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
    except Exception as e:
        print(f"创建源域数据集时出错: {e}")
        source_dataloader = None
    
    # 创建目标域数据集
    print(f"创建目标域数据集...目标建筑: {target_building}")
    try:
        target_dataset = ChronosDataset(
            electricity_data=target_electricity,
            weather_data=test_weather,
            buildings=[target_building],
            building_to_weather=building_to_weather,
            building_to_category=building_to_category,
            categories=categories,
            sequence_length=sequence_length,
            forecast_horizon=forecast_horizon
        )
        
        target_dataloader = DataLoader(
            target_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
    except Exception as e:
        print(f"创建目标域数据集时出错: {e}")
        target_dataloader = None
    
    # 打印数据集信息
    source_size = len(source_dataset) if source_dataloader is not None else 0
    target_size = len(target_dataset) if target_dataloader is not None else 0
    print(f"源域数据集大小: {source_size}, 目标域数据集大小: {target_size}")
    print(f"源域建筑: {source_domain_buildings}")
    print(f"目标域建筑: {target_building}")
    
    return source_dataloader, target_dataloader, categories