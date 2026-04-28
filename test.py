import pandas as pd
driver_features = pd.read_csv('./data/driver_features.csv')
n_driver = len(driver_features)

import pandas as pd
import numpy as np

# 读取数据
race_data = pd.read_csv('data/f1_multi_season_results.csv')  # 你的比赛数据
teams_info = pd.read_csv('data/teams_info.csv')
driver_features = pd.read_csv('data/driver_features.csv')

# 1. 添加 TrackType（根据赛道名称）
track_types = {
    'Australia': 'balanced', 'China': 'high_speed', 'Japan': 'technical',
    'Bahrain': 'high_speed', 'Saudi Arabia': 'high_speed', 'Miami': 'high_speed',
    'Emilia Romagna': 'technical', 'Monaco': 'technical', 'Spain': 'balanced',
    'Canada': 'balanced', 'Austria': 'technical', 'Great Britain': 'balanced',
    'Belgium': 'high_speed', 'Hungary': 'technical', 'Netherlands': 'technical',
    'Italy': 'high_speed', 'Azerbaijan': 'high_speed', 'Singapore': 'technical',
    'United States': 'balanced', 'Mexico': 'balanced', 'Brazil': 'balanced',
    'Las Vegas': 'high_speed', 'Qatar': 'high_speed', 'Abu Dhabi': 'balanced',
}

if 'TrackType' not in race_data.columns:
    race_data['TrackType'] = race_data['Race'].map(track_types)

# 2. 添加 TrackTypeIdx
track_type_map = {'high_speed': 0, 'balanced': 1, 'technical': 2}
race_data['TrackTypeIdx'] = race_data['TrackType'].map(track_type_map)

# 3. 从 teams_info 添加 TierCode
# 创建车队到Tier的映射
team_tier_map = dict(zip(teams_info['Team'], teams_info['Tier']))
race_data['TierCode'] = race_data['Team'].map(team_tier_map)

# 4. 添加 DriverIdx（为每个车手分配唯一索引）
unique_drivers = race_data['Driver'].unique()
driver_to_idx = {driver: idx for idx, driver in enumerate(unique_drivers)}
race_data['DriverIdx'] = race_data['Driver'].map(driver_to_idx)
n_driver = len(unique_drivers)

# 5. 添加 EffectivePosition（未完赛=21）
race_data['EffectivePosition'] = race_data['Position'].copy()
race_data.loc[race_data['Status'] != 'Finished', 'EffectivePosition'] = 21.0

# 6. 从 driver_features 获取 Recent5Avg（如果存在）
# 创建车手到Recent5Avg的映射
driver_recent5_map = dict(zip(driver_features['Driver'], driver_features['Recent5Avg']))

# 但是我们需要动态计算每场比赛时的Recent5Avg，而不是使用全赛季的
# 所以还是按时间顺序计算
race_data = race_data.sort_values(['Driver', 'GlobalRound'])
race_data['Recent5Avg'] = race_data.groupby('Driver')['EffectivePosition'].transform(
    lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
)
# 第一场比赛用车手的平均值填充（从driver_features获取）
driver_avg_map = dict(zip(driver_features['Driver'], driver_features['AvgPosition']))
race_data['Recent5Avg'] = race_data.apply(
    lambda row: driver_avg_map.get(row['Driver'], race_data['EffectivePosition'].mean()) 
    if pd.isna(row['Recent5Avg']) else row['Recent5Avg'],
    axis=1
)

# 7. 计算 CumDNFRate（累计未完赛率）
race_data['IsDNF'] = (race_data['Status'] != 'Finished').astype(int)
race_data['CumDNFRate'] = race_data.groupby('Driver')['IsDNF'].transform(
    lambda x: x.shift(1).expanding().mean()
)
# 第一场比赛用车手的DNFRate填充（从driver_features获取）
driver_dnf_map = dict(zip(driver_features['Driver'], driver_features['DNFRate']))
race_data['CumDNFRate'] = race_data.apply(
    lambda row: driver_dnf_map.get(row['Driver'], 0) 
    if pd.isna(row['CumDNFRate']) else row['CumDNFRate'],
    axis=1
)

# 8. 确保所有必要的列都存在
required_columns = [
    'Driver', 'Team', 'Race', 'TrackType', 'TrackTypeIdx', 
    'TierCode', 'DriverIdx', 'GridPosition', 'QualifyingPosition',
    'Position', 'EffectivePosition', 'Recent5Avg', 'CumDNFRate', 
    'Status', 'GlobalRound', 'Season'
]

# 保存处理后的数据
race_data.to_csv('data/f1_race_data_cleaned.csv', index=False)

print(f"处理完成！")
print(f"车手数量 (n_driver): {n_driver}")
print(f"总比赛场次: {len(race_data)}")
print(f"\n数据预览:")
print(race_data[['Driver', 'Team', 'Race', 'TierCode', 'TrackTypeIdx', 
                  'EffectivePosition', 'Recent5Avg', 'CumDNFRate']].head(10))

# 检查缺失值
print(f"\n缺失值检查:")
print(race_data[['TierCode', 'TrackTypeIdx', 'Recent5Avg', 'CumDNFRate']].isnull().sum())

# 输出车队tier分布
print(f"\n车队Tier分布:")
print(race_data.groupby(['Team', 'TierCode']).size().sort_values(ascending=False))


import pymc as pm

with pm.Model() as f1_final_model:

    # level 1 priori team
    mu_tier = [-12,-5,0]
    sigma_tier = 2
    beta_team = pm.Normal('beta_team',mu=mu_tier,sigma=sigma_tier,shape=3)
    # level 2 driver ability
    sigma_driver = pm.HalfNormal('sigma_driver',2.0)
    gamma_driver = pm.Normal('gamma_driver',mu =0,sigma = sigma_driver, shape = n_driver)
    # level 3 settings
    alpha = pm.Normal('alpha', mu=10.6, sigma=2)
    # delta_track = pm.Normal('delta_track', mu=0, sigma=3.5)
    eta_grid = pm.Normal('eta_grid', mu=0, sigma=0.5)
    epsilon_trend = pm.Normal('epsilon_trend', mu=0, sigma=3.5)
    zeta_dnf = pm.HalfNormal('zeta_dnf', 3.5)
    # sigma_race = pm.HalfNormal('sigma_race', 1.0) 

    # Track type effects (3 types: high_speed, balanced, technical)
    delta_track = pm.Normal('delta_track', mu=0, sigma=2, shape=3)
    sigma_race = pm.HalfNormal('sigma_race', 1.0)
    xi_quali = pm.Normal('xi_quali', mu=0, sigma=0.8)  

    track_type_map = {'high_speed': 0, 'balanced': 1, 'technical': 2}
    race_data['TrackTypeIdx'] = race_data['TrackType'].map(track_type_map)

    # likelyhood
    mu_position = (
        alpha +
        beta_team[race_data['TierCode'].values] +
        gamma_driver[race_data['DriverIdx'].values] +  # 使用 driver index!
        eta_grid * race_data['GridPosition'].values +
        # xi_quali * race_data['QualifyingPosition'].values +
        delta_track[race_data['TrackTypeIdx'].values] +  # Track type effect
        epsilon_trend * race_data['Recent5Avg'].values +
        zeta_dnf * race_data['CumDNFRate'].values
    )
    y_obs = pm.Normal(
        'y_obs', 
        mu=mu_position, 
        sigma=sigma_race,
        observed=race_data['EffectivePosition'].values  # 440 observations!
        # observed=race_data['QualifyingPosition'].values  # 440 observations!
    )


# 这个注释下的代码要保留，但是同样要修改

# ========================================
# test simulation 
# find the best randomseeds
# ==========================================

# # get most usefull random seed
# import arviz as az
# import time

# # 测试 5 个不同的 seeds
# seeds = [2020,2022,2024,2026,2030]
# results = []

# for seed in seeds:
#     print(f"\n{'='*60}")
#     print(f"Testing seed = {seed}")
#     print(f"{'='*60}")
    
#     start = time.time()
    
#     with f1_final_model:
#         trace_test = pm.sample(
#             draws=4000,
#             tune=2000,
#             chains=2,
#             cores=2,
#             target_accept=0.99,
#             random_seed=seed,
#             progressbar=False  # 关闭进度条，输出更清晰
#         )
    
#     elapsed = time.time() - start
    
#     # 诊断
#     rhat = az.rhat(trace_test)
#     ess = az.ess(trace_test)
    
#     max_rhat = max([float(rhat[var].max().values) for var in rhat.data_vars])
#     min_ess = min([float(ess[var].min().values) for var in ess.data_vars])
#     divergences = int(trace_test.sample_stats.diverging.sum().values)
    
#     results.append({
#         'seed': seed,
#         'max_rhat': max_rhat,
#         'min_ess': min_ess,
#         'divergences': divergences,
#         'time': elapsed
#     })
    
#     print(f"Max R-hat: {max_rhat:.4f}")
#     print(f"Min ESS: {min_ess:.0f}")
#     print(f"Divergences: {divergences}")
#     print(f"Time: {elapsed:.1f}s")

# # 总结
# import pandas as pd
# df_results = pd.DataFrame(results)
# print(f"\n{'='*60}")
# print("SUMMARY ACROSS SEEDS")
# print(f"{'='*60}")
# print(df_results)

# print(f"\nESS statistics:")
# print(f"  Mean: {df_results['min_ess'].mean():.0f}")
# print(f"  Min: {df_results['min_ess'].min():.0f}")
# print(f"  Max: {df_results['min_ess'].max():.0f}")

# # 选择最好的 seed
# best_seed = df_results.loc[df_results['min_ess'].idxmax(), 'seed']
# print(f"\n✓ Best seed: {best_seed} (ESS = {df_results['min_ess'].max():.0f})")


with f1_final_model:
    trace_test = pm.sample(4000, tune=2000, random_seed=2024, chains=4,cores=4, target_accept=0.998,progressbar=True)
    trace_test.to_netcdf('./model/f1_trace.nc')
    print("✓ Trace saved to './model/f1_trace.nc'")


import arviz as az
import numpy as np

# 加载 trace
trace = az.from_netcdf('./model/f1_trace.nc')

print("="*60)
print("QUICK DIAGNOSTICS")
print("="*60)

# 1. R-hat (修正版本)
rhat = az.rhat(trace)
# 提取所有变量的最大 R-hat
max_rhat_values = []
for var in rhat.data_vars:
    max_rhat_values.append(float(rhat[var].max().values))
max_rhat = max(max_rhat_values)

print(f"\nMax R-hat: {max_rhat:.4f}")

if max_rhat > 1.01:
    print("⚠️ WARNING: Chains have NOT converged!")
else:
    print("✓ Chains converged")

# 2. ESS (修正版本)
ess = az.ess(trace)
min_ess_values = []
for var in ess.data_vars:
    min_ess_values.append(float(ess[var].min().values))
min_ess = min(min_ess_values)

print(f"Min ESS: {min_ess:.0f}")

if min_ess < 400:
    print("⚠️ WARNING: Low ESS")
else:
    print("✓ Sufficient ESS")

# 3. Divergences (修正版本)
divergences = int(trace.sample_stats.diverging.sum().values)
print(f"Divergences: {divergences}")

if divergences > 0:
    print(f"⚠️ {divergences} divergent transitions")
else:
    print("✓ No divergences")

# 总结
print("\n" + "="*60)
if max_rhat < 1.01 and min_ess > 400:
    print("✓ MODEL IS USABLE (despite divergence)")
    print("You can proceed with analysis!")
elif max_rhat > 1.01:
    print("❌ NEED TO RESAMPLE with higher target_accept")
elif min_ess < 100:
    print("❌ NEED MORE SAMPLES")
else:
    print("⚠️ MODEL IS OKAY but could be better")

# 运行这个来确认统计有效性
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
import pymc as pm

race_data = pd.read_csv('./data/f1_race_data_cleaned.csv')

with f1_final_model:
    posterior_pred = pm.sample_posterior_predictive(trace, random_seed=42)

y_pred = posterior_pred.posterior_predictive['y_obs'].values
y_obs = race_data['EffectivePosition'].values

y_pred_mean = y_pred.mean(axis=(0, 1))

r2 = r2_score(y_obs, y_pred_mean)
mae = mean_absolute_error(y_obs, y_pred_mean)

print(f"R² = {r2:.3f}")
print(f"MAE = {mae:.2f} positions")