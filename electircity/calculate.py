import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import math
import scipy.stats as stats

def calculate_mape(y_true, y_pred):
    mask = np.abs(y_true) > 1e-4
    if not np.any(mask):
        return np.nan  
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def calculate_uncertainty_metrics(y_pred, y_true, pred_lower, pred_upper, confidence=0.95):
    in_interval = (y_true >= pred_lower) & (y_true <= pred_upper)
    picp = np.mean(in_interval)
    
    mpiw = np.mean(pred_upper - pred_lower)
    
    y_range = np.max(y_true) - np.min(y_true)
    nmpiw = mpiw / y_range if y_range > 0 else np.nan
    
    calibration_error = np.abs(picp - confidence)
    
    if picp < confidence:
        coverage_ratio = picp / confidence
        coverage_penalty = np.exp(-2 * (1 - coverage_ratio))
        uqs = coverage_penalty * (1 - nmpiw) if nmpiw > 0 else np.nan
    else:
        excess_coverage = max(0, picp - confidence)
        excess_penalty = 1.0 - min(0.1, excess_coverage)
        uqs = excess_penalty * (1 - nmpiw) if nmpiw > 0 else np.nan
    
    sharpness = 1 - nmpiw if nmpiw > 0 else np.nan
    interval_efficiency = picp / nmpiw if nmpiw > 0 else np.nan
    
    return {
        "PICP": picp * 100,  
        "MPIW": mpiw,
        "NMPIW": nmpiw,
        "calibration_error": calibration_error * 100,  
        "UQS": uqs,
        "sharpness": sharpness,
        "interval_efficiency": interval_efficiency
    }

def calculate_metrics(predictions, real_values, model_name="Model", baseline_predictions=None, 
                     lower_bounds=None, upper_bounds=None, confidence=0.95):
    predictions = np.array(predictions)
    real_values = np.array(real_values)
    
    mean_real = np.mean(real_values)
    
    mape = calculate_mape(real_values, predictions)
    
    rmsd = np.sqrt(mean_squared_error(real_values, predictions))
    
    # 计算 SD（真实值、预测值标准差）
    sd_real = np.std(real_values)
    sd_pred = np.std(predictions)
    
    # 计算 CC（相关系数）
    cc = np.corrcoef(predictions.flatten(), real_values.flatten())[0, 1]
    
    # 计算 R2（决定系数）
    r2 = r2_score(real_values, predictions)
    
    # 计算 CV-RMSE（RMSD 与真实值均值的比值）
    cv_rmse = (rmsd / mean_real) * 100 if mean_real != 0 else np.nan
    
    pir = None
    if baseline_predictions is not None:
        baseline_predictions = np.array(baseline_predictions)
        rmsd_baseline = np.sqrt(mean_squared_error(real_values, baseline_predictions))
        pir = (rmsd_baseline - rmsd) / rmsd_baseline * 100
    
    uncertainty_metrics = {}
    if lower_bounds is not None and upper_bounds is not None:
        lower_bounds = np.array(lower_bounds)
        upper_bounds = np.array(upper_bounds)
        uncertainty_metrics = calculate_uncertainty_metrics(
            predictions, real_values, lower_bounds, upper_bounds, confidence
        )
    
    metrics = {
        "Model": model_name,
        "MAPE": mape,
        "RMSD": rmsd,
        "CV-RMSE": cv_rmse,  # 新增 CV-RMSE
        "SD_real": sd_real,  # 新增真实值标准差
        "SD_pred": sd_pred,  # 新增预测值标准差
        "CC": cc,
        "R2": r2,
        "PIR": pir
    }
    
    if uncertainty_metrics:
        metrics.update(uncertainty_metrics)
    
    return metrics

def print_metrics_table(metrics_list):
    # 主指标表格：包含 SD、CV-RMSE、CC
    print("{:<15} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
        "Model", "MAPE(%)", "RMSD", "CV-RMSE", 
        "SD_real", "SD_pred", "CC", "R2", "PIR(%)"
    ))
    print("-" * 100)
    
    for metrics in metrics_list:
        pir_str = f"{metrics['PIR']:.2f}" if metrics['PIR'] is not None else "N/A"
        print("{:<15} {:<10.2f} {:<10.2f} {:<10.2f} {:<10.2f} {:<10.2f} {:<10.2f} {:<10.2f} {:<10}".format(
            metrics["Model"], 
            metrics["MAPE"], 
            metrics["RMSD"], 
            metrics["CV-RMSE"],
            metrics["SD_real"],
            metrics["SD_pred"],
            metrics["CC"],
            metrics["R2"],
            pir_str
        ))
    
    # 不确定性指标表格（按需打印）
    has_uncertainty = any("PICP" in metrics for metrics in metrics_list)
    if has_uncertainty:
        print("\n不确定性评估指标:")
        print("{:<15} {:<10} {:<15} {:<15} {:<10} {:<10} {:<10}".format(
            "Model", "PICP(%)", "校准误差(%)", "NMPIW", 
            "UQS", "sharpness", "interval_efficiency"
        ))
        print("-" * 100)
        
        for metrics in metrics_list:
            if "PICP" in metrics:
                print("{:<15} {:<10.2f} {:<15.2f} {:<15.4f} {:<10.4f} {:<10.4f} {:<10.4f}".format(
                    metrics["Model"], 
                    metrics["PICP"], 
                    metrics["calibration_error"], 
                    metrics["NMPIW"],
                    metrics.get("UQS", float('nan')),
                    metrics.get("sharpness", float('nan')),
                    metrics.get("interval_efficiency", float('nan'))
                ))

