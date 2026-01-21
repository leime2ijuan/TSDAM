"""
dataloader.py

Time series data loader - supports data scarcity scenarios
- Training buildings use full data
- Target buildings use scarce data (mild/heavy/extreme)
- Supports train/validation time split
- Special handling for CC category
"""

import pandas as pd
import numpy as np
import os
import json
import torch
import random
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from typing import Tuple, List, Dict, Optional


def custom_collate_fn(batch):
    """
    Custom collate function to handle batches with varying number of buildings
    
    Input format:
        batch: list of (features, targets, category, category_onehot)
        - features: [num_buildings, seq_len, input_dim]
        - targets: [num_buildings, forecast_horizon]
    
    Output format:
        - features_batch: [total_buildings, 1, seq_len, input_dim]
        - targets_batch: [total_buildings, 1, forecast_horizon]
        - category_batch: str (same for all samples in batch)
        - category_onehot_batch: [total_buildings, num_categories]
    
    Logic:
        Flatten all buildings across samples into the batch dimension,
        treating each building as an independent sample (building_dim = 1)
    """
    features_list = []
    targets_list = []
    categories_list = []
    category_onehot_list = []
    
    for item in batch:
        features, targets, category, category_onehot = item
        
        num_buildings = features.shape[0]
        
        for b in range(num_buildings):
            features_list.append(features[b:b+1, :, :])      # [1, seq_len, input_dim]
            targets_list.append(targets[b:b+1, :])          # [1, forecast_horizon]
            categories_list.append(category)
            category_onehot_list.append(category_onehot)
    
    features_batch = torch.cat(features_list, dim=0)                    # [total_buildings, seq_len, input_dim]
    targets_batch = torch.cat(targets_list, dim=0)                      # [total_buildings, forecast_horizon]
    
    features_batch = features_batch.unsqueeze(1)                        # [total_buildings, 1, seq_len, input_dim]
    targets_batch = targets_batch.unsqueeze(1)                          # [total_buildings, 1, forecast_horizon]
    
    category_batch = categories_list[0]                                 # All buildings in batch share the same category
    category_onehot_batch = torch.stack(category_onehot_list, dim=0)    # [total_buildings, num_categories]
    
    return features_batch, targets_batch, category_batch, category_onehot_batch


class ChronosDataset(Dataset):
    """
    Chronos Time Series Dataset
    
    Supports multi-building, multi-feature time series data including electricity and weather.
    
    Data format:
        - Input:  [buildings, sequence_length, features]
        - Target: [buildings, forecast_horizon]
        - Features: [electricity, temperature, dew_point, pressure, wind_direction, wind_speed]
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
        
        # Find common timestamps across electricity and weather data
        common_timestamps = set(electricity_data.index)
        for building in buildings:
            station = building_to_weather[building]
            if station in weather_data and weather_data[station] is not None:
                common_timestamps &= set(weather_data[station].index)
        
        self.timestamps = sorted(list(common_timestamps))
        
        # Apply time range restriction
        if time_range != 'all':
            split_idx = int(len(self.timestamps) * 0.8)
            if time_range == 'first_80_percent':
                self.timestamps = self.timestamps[:split_idx]
            elif time_range == 'last_20_percent':
                self.timestamps = self.timestamps[split_idx:]
        
        # Check for missing values
        self.has_missing = self.electricity_data.isna().any().any()
        if self.has_missing:
            missing_rate = self.electricity_data.isna().sum().sum() / \
                          (len(self.electricity_data) * len(self.electricity_data.columns)) * 100
            print(f"  Warning: Data contains missing values (rate: {missing_rate:.2f}%)")
        
        print(f"  Dataset: {len(buildings)} buildings, "
              f"{len(self.timestamps) - sequence_length - forecast_horizon} valid samples"
              f"{f' ({time_range})' if time_range != 'all' else ''}")
    
    def __len__(self) -> int:
        return max(1, len(self.timestamps) - self.sequence_length - self.forecast_horizon)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str, torch.Tensor]:
        # Compute time window
        input_start_idx = idx
        input_end_idx = idx + self.sequence_length
        target_start_idx = input_end_idx
        target_end_idx = target_start_idx + self.forecast_horizon
        
        # Boundary handling
        if target_end_idx > len(self.timestamps):
            target_end_idx = len(self.timestamps)
            target_start_idx = target_end_idx - self.forecast_horizon
            input_end_idx = target_start_idx
            input_start_idx = input_end_idx - self.sequence_length
        
        input_timestamps = self.timestamps[input_start_idx:input_end_idx]
        target_timestamps = self.timestamps[target_start_idx:target_end_idx]
        
        all_features = []
        all_targets = []
        
        for building in self.buildings:
            try:
                # Electricity
                electricity_input = self.electricity_data.loc[input_timestamps, building].values
                electricity_target = self.electricity_data.loc[target_timestamps, building].values
                
                # Weather
                weather_station = self.building_to_weather[building]
                weather_features = 5
                weather_input = np.zeros((len(input_timestamps), weather_features))
                
                if weather_station in self.weather_data and self.weather_data[weather_station] is not None:
                    weather_df = self.weather_data[weather_station]
                    weather_cols = [col for col in weather_df.columns if col != 'site_id'][:weather_features]
                    
                    for i, ts in enumerate(input_timestamps):
                        if ts in weather_df.index:
                            for j, col in enumerate(weather_cols):
                                if j < weather_features:
                                    try:
                                        val = weather_df.loc[ts, col]
                                        if isinstance(val, (int, float)):
                                            weather_input[i, j] = val
                                    except:
                                        pass
                
                # Handle missing values
                if self.handle_missing == 'forward_fill':
                    electricity_input = pd.Series(electricity_input).ffill().bfill().fillna(0).values
                    electricity_target = pd.Series(electricity_target).ffill().bfill().fillna(0).values
                    weather_input = pd.DataFrame(weather_input).ffill().bfill().fillna(0).values
                elif self.handle_missing == 'zero_fill':
                    electricity_input = np.nan_to_num(electricity_input, nan=0.0)
                    electricity_target = np.nan_to_num(electricity_target, nan=0.0)
                    weather_input = np.nan_to_num(weather_input, nan=0.0)
                
                electricity_input = electricity_input.reshape(-1, 1)
                
                # Fix length if necessary
                if len(electricity_input) != self.sequence_length:
                    electricity_input = np.resize(electricity_input, (self.sequence_length, 1))
                
                if len(weather_input) != self.sequence_length:
                    if len(weather_input) > self.sequence_length:
                        weather_input = weather_input[:self.sequence_length]
                    else:
                        padding = np.zeros((self.sequence_length - len(weather_input), weather_features))
                        weather_input = np.vstack([weather_input, padding])
                
                combined_features = np.concatenate([electricity_input, weather_input], axis=1)
                
                all_features.append(combined_features)
                all_targets.append(electricity_target)
                
            except Exception as e:
                print(f"Warning: Error processing building {building}: {str(e)}")
                combined_features = np.zeros((self.sequence_length, 6))
                electricity_target = np.zeros(self.forecast_horizon)
                all_features.append(combined_features)
                all_targets.append(electricity_target)
        
        features_tensor = torch.tensor(np.array(all_features), dtype=torch.float32)
        targets_tensor = torch.tensor(np.array(all_targets), dtype=torch.float32)
        
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
    Load source domain data loaders (only buildings labeled 'train' in train_test_labels.json)
    """
    print("="*70)
    print("Loading Source Domain Training Data")
    print("="*70)
    print("Description:")
    print("  • Only uses buildings marked as 'train' in train_test_labels.json")
    print("  • Full complete data (no artificial missing values)")
    print(f"  • Temporal split: first {(1-val_ratio)*100:.0f}% for training, last {val_ratio*100:.0f}% for validation")
    print("="*70)
    
    # Load config files
    with open('weather_labels.json', 'r') as f:
        weather_labels = json.load(f)
    with open('train_test_labels.json', 'r') as f:
        train_test_labels = json.load(f)
    
    # Build mappings
    building_to_weather = {b["building_id"]: b["weather_station"] for b in weather_labels["buildings"]}
    building_to_category = {}
    categories = list(train_test_labels.keys())
    
    for cat in categories:
        for building in train_test_labels[cat].get("train", []):
            building_to_category[building] = cat
        for building in train_test_labels[cat].get("test", []):
            building_to_category[building] = cat
    
    # Collect all training buildings
    all_train_buildings = []
    for cat in categories:
        all_train_buildings.extend(train_test_labels[cat].get("train", []))
    
    print(f"\nBuilding Groups:")
    print(f"  Training buildings: {len(all_train_buildings)}")
    
    # Load electricity data
    print(f"\nLoading electricity data...")
    full_electricity = pd.read_csv('data/electricity_train_buildings_only.csv')
    full_electricity['timestamp'] = pd.to_datetime(full_electricity['timestamp'])
    full_electricity.set_index('timestamp', inplace=True)
    train_electricity = full_electricity[all_train_buildings].copy()
    print(f"  Complete electricity data: {full_electricity.shape}")
    print(f"  Training electricity data: {train_electricity.shape}")
    
    # Load weather data
    print(f"\nLoading weather data...")
    def load_weather_file(station: str):
        path = f'data/weather_data/{station}_processed.csv'
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            if 'site_id' in df.columns:
                df = df.drop(columns=['site_id'])
            return df
        else:
            print(f"    Warning: Not found: {path}")
            return None
    
    train_weather_stations = {building_to_weather[b] for b in all_train_buildings}
    train_weather = {}
    for station in train_weather_stations:
        data = load_weather_file(station)
        train_weather[station] = data
        if data is not None:
            print(f"    {station}: {data.shape}")
    
    # Create datasets
    print(f"\nCreating data loaders...")
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
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=True, collate_fn=custom_collate_fn)
    
    print(f"\n" + "="*70)
    print(f"Source domain data loading completed!")
    print("="*70)
    print(f"Training samples: {len(train_dataset)} (first 80% time)")
    print(f"Validation samples: {len(val_dataset)} (last 20% time)")
    print(f"Batch size: {batch_size}")
    print(f"Missing handling: {handle_missing}")
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
    Load transfer learning data loaders
    
    Training data:
    - Full data from source buildings (labeled 'train')
    - Plus scarce training portion (first 80%) of target buildings
    
    Test data:
    - Scarce test portion (last 20%, clean) of target buildings
    
    Special case - CC category:
    - If category is CC (no training buildings), randomly pick 1 building from each of the other 5 categories
    """
    print("="*70)
    print(f"Loading Transfer Learning Data - Category: {category}, Scarcity: {data_shortage.upper()}")
    print("="*70)
    print("Description:")
    print("  • Training: source buildings (full) + target scarce training (first 80%)")
    print("  • Testing: target scarce test (last 20%, clean)")
    if category == 'CC':
        print("  • Special handling: CC has no training buildings → randomly select 1 from each of other 5 categories")
    print("="*70)
    
    # Load configs
    with open('weather_labels.json', 'r') as f:
        weather_labels = json.load(f)
    with open('train_test_labels.json', 'r') as f:
        train_test_labels = json.load(f)
    
    building_to_weather = {b["building_id"]: b["weather_station"] for b in weather_labels["buildings"]}
    building_to_category = {}
    categories = list(train_test_labels.keys())
    
    for cat in categories:
        for b in train_test_labels[cat].get("train", []):
            building_to_category[b] = cat
        for b in train_test_labels[cat].get("test", []):
            building_to_category[b] = cat
    
    # Get source and target buildings
    if category == 'CC':
        train_buildings = []
        for cat in categories:
            if cat != 'CC' and train_test_labels[cat].get("train"):
                selected = random.choice(train_test_labels[cat]["train"])
                train_buildings.append(selected)
                print(f"  Randomly selected from {cat}: {selected}")
    else:
        train_buildings = train_test_labels[category].get("train", [])
    
    test_buildings = train_test_labels[category].get("test", [])
    if not test_buildings:
        print(f"Error: Category {category} has no test buildings")
        return None, None, categories
    
    print(f"\nBuilding Groups:")
    print(f"  Training buildings: {len(train_buildings)} → {train_buildings}")
    print(f"  Target buildings: {len(test_buildings)} → {test_buildings}")
    
    # Load electricity data
    print(f"\nLoading electricity data...")
    full_electricity = pd.read_csv('data/electricity_train_buildings_only.csv')
    full_electricity['timestamp'] = pd.to_datetime(full_electricity['timestamp'])
    full_electricity.set_index('timestamp', inplace=True)
    
    scarcity_electricity = pd.read_csv(f'data/target_scarcity_data/scarcity_data/data_{data_shortage}_scarcity.csv')
    scarcity_electricity['timestamp'] = pd.to_datetime(scarcity_electricity['timestamp'])
    scarcity_electricity.set_index('timestamp', inplace=True)
    
    split_idx = int(len(scarcity_electricity) * 0.8)
    train_timestamps = scarcity_electricity.index[:split_idx]
    test_timestamps = scarcity_electricity.index[split_idx:]
    
    train_scarcity_electricity = scarcity_electricity.loc[train_timestamps]
    test_scarcity_electricity = scarcity_electricity.loc[test_timestamps]
    
    print(f"  Full electricity: {full_electricity.shape}")
    print(f"  Scarce electricity: {scarcity_electricity.shape}")
    print(f"  Scarce train portion: {train_scarcity_electricity.shape} (first 80%)")
    print(f"  Scarce test portion: {test_scarcity_electricity.shape} (last 20%)")
    
    # Load weather data
    print(f"\nLoading weather data...")
    def load_weather_file(station: str, is_complete: bool = True):
        if is_complete:
            path = f'data/weather_data/{station}_processed.csv'
        else:
            path = f'data/target_scarcity_data/scarcity_data/weather/{station}_{data_shortage}_scarcity.csv'
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            if 'site_id' in df.columns:
                df.drop(columns=['site_id'], inplace=True)
            return df
        else:
            print(f"    Warning: Not found: {path}")
            return None
    
    train_weather = {}
    if train_buildings:
        stations = {building_to_weather[b] for b in train_buildings}
        for station in stations:
            train_weather[station] = load_weather_file(station, is_complete=True)
    
    test_weather = {}
    test_stations = {building_to_weather[b] for b in test_buildings}
    for station in test_stations:
        data = load_weather_file(station, is_complete=False)
        test_weather[station] = data
        if data is not None:
            missing_rate = data.isna().sum().sum() / (data.size) * 100
            print(f"    {station}: {data.shape} (missing: {missing_rate:.2f}%)")
    
    # Create datasets
    print(f"\nCreating data loaders...")
    train_buildings_dataset = None
    if train_buildings:
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
    
    # Combine training datasets
    if train_buildings_dataset is not None:
        combined_train_dataset = ConcatDataset([train_buildings_dataset, target_train_dataset])
    else:
        combined_train_dataset = target_train_dataset
    
    train_loader = DataLoader(combined_train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True, collate_fn=custom_collate_fn)
    test_loader = DataLoader(target_test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=0, pin_memory=True, collate_fn=custom_collate_fn)
    
    print(f"\n" + "="*70)
    print(f"Transfer learning data for {category} loaded successfully!")
    print("="*70)
    print(f"Training samples: {len(combined_train_dataset)}")
    if train_buildings_dataset is not None:
        print(f"  - Source buildings: {len(train_buildings_dataset)} samples")
    print(f"  - Target scarce train: {len(target_train_dataset)} samples")
    print(f"Test samples: {len(target_test_dataset)}")
    print(f"Batch size: {batch_size}")
    print(f"Missing handling: {handle_missing}")
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
    [New function] Load pure target domain data loaders (for Pure BiLSTM baseline)
    
    Features:
    - Only uses scarce target domain data
    - Strictly excludes any source domain full data
    - Used to evaluate performance when training only on scarce data (no transfer)
    """
    print("="*70)
    print(f"Loading Pure Target Domain Data (No Transfer) - Category: {category}, Scarcity: {data_shortage.upper()}")
    print("="*70)
    
    with open('weather_labels.json', 'r') as f:
        weather_labels = json.load(f)
    with open('train_test_labels.json', 'r') as f:
        train_test_labels = json.load(f)
    
    building_to_weather = {b["building_id"]: b["weather_station"] for b in weather_labels["buildings"]}
    building_to_category = {}
    categories = list(train_test_labels.keys())
    
    test_buildings = train_test_labels[category].get("test", [])
    for b in test_buildings:
        building_to_category[b] = category
    
    if not test_buildings:
        print(f"Error: Category {category} has no test buildings")
        return None, None, categories
    
    print(f"  Target buildings (only these): {len(test_buildings)} → {test_buildings}")
    
    # Load scarce electricity
    scarcity_path = f'data/target_scarcity_data/scarcity_data/data_{data_shortage}_scarcity.csv'
    if not os.path.exists(scarcity_path):
        print(f"Error: File not found: {scarcity_path}")
        return None, None, categories
    
    scarcity_electricity = pd.read_csv(scarcity_path)
    scarcity_electricity['timestamp'] = pd.to_datetime(scarcity_electricity['timestamp'])
    scarcity_electricity.set_index('timestamp', inplace=True)
    
    split_idx = int(len(scarcity_electricity) * 0.8)
    timestamps = scarcity_electricity.index.tolist()
    train_timestamps = timestamps[:split_idx]
    test_timestamps = timestamps[split_idx:]
    
    # Load scarce weather
    print(f"\nLoading scarce weather data ({data_shortage})...")
    weather_data = {}
    stations = {building_to_weather[b] for b in test_buildings}
    for station in stations:
        path = f'data/target_scarcity_data/scarcity_data/weather/{station}_{data_shortage}_scarcity.csv'
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            if 'site_id' in df.columns:
                df.drop(columns=['site_id'], inplace=True)
            weather_data[station] = df
        else:
            print(f"    Warning: Weather file not found: {path}")
    
    # Create datasets
    print(f"\nCreating pure target domain datasets...")
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
        time_range='all'
    )
    
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
    
    train_loader = DataLoader(target_train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True, collate_fn=custom_collate_fn)
    test_loader = DataLoader(target_test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=0, pin_memory=True, collate_fn=custom_collate_fn)
    
    print(f"Pure target domain data loaded: Train {len(target_train_dataset)} samples, Test {len(target_test_dataset)} samples")
    return train_loader, test_loader, categories
