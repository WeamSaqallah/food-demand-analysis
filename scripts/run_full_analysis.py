import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Set styling for publication-grade charts
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

base_dir = os.path.abspath('food-demand-analysis')
raw_dir = os.path.join(base_dir, 'data', 'raw')
fig_dir = os.path.join(base_dir, 'reports', 'figures')
os.makedirs(fig_dir, exist_ok=True)

# 1. Load Data
print("Loading raw data...")
train = pd.read_csv(os.path.join(raw_dir, 'train.csv'))
centers = pd.read_csv(os.path.join(raw_dir, 'fulfilment_center_info.csv'))
meals = pd.read_csv(os.path.join(raw_dir, 'meal_info.csv'))

# 2. Merge Data
df = train.merge(centers, on='center_id', how='left').merge(meals, on='meal_id', how='left')

# 3. Feature Engineering
df['revenue'] = df['checkout_price'] * df['num_orders']
df['discount_amount'] = np.maximum(0, df['base_price'] - df['checkout_price'])
df['discount_rate'] = np.maximum(0, (df['base_price'] - df['checkout_price']) / df['base_price'])
df['has_discount'] = (df['base_price'] > df['checkout_price']).astype(int)
df['has_promo'] = ((df['emailer_for_promotion'] == 1) | (df['homepage_featured'] == 1)).astype(int)

# -------------------------------------------------------------
# Q1: Weekly Total Demand Trend
# -------------------------------------------------------------
weekly = df.groupby('week').agg(
    total_orders=('num_orders', 'sum'),
    total_revenue=('revenue', 'sum'),
    active_centers=('center_id', 'nunique'),
    active_meals=('meal_id', 'nunique')
).reset_index()

overall_total_orders = df['num_orders'].sum()
overall_total_revenue = df['revenue'].sum()
weekly_mean_orders = weekly['total_orders'].mean()
weekly_std_orders = weekly['total_orders'].std()
weekly_min_orders = weekly['total_orders'].min()
weekly_max_orders = weekly['total_orders'].max()
min_week = weekly.loc[weekly['total_orders'].idxmin(), 'week']
max_week = weekly.loc[weekly['total_orders'].idxmax(), 'week']

print(f"Total Orders: {overall_total_orders:,}")
print(f"Total Revenue: {overall_total_revenue:,.2f}")
print(f"Weekly Average Orders: {weekly_mean_orders:,.1f} +/- {weekly_std_orders:,.1f}")
print(f"Min Week: Week {min_week} ({weekly_min_orders:,}), Max Week: Week {max_week} ({weekly_max_orders:,})")

# Figure 1: Weekly Demand Trend
plt.figure(figsize=(12, 5), dpi=300)
plt.plot(weekly['week'], weekly['total_orders'] / 1e3, color='#1f77b4', lw=2, label='Weekly Demand (in Thousands)')
# 4-week rolling average
weekly['rolling_4w'] = weekly['total_orders'].rolling(window=4).mean()
plt.plot(weekly['week'], weekly['rolling_4w'] / 1e3, color='#ff7f0e', lw=2.5, linestyle='--', label='4-Week Moving Average')
plt.title('Weekly Demand Fluctuation Around 824K Orders Mean with Peak at Week 48 (1,009K Orders)', fontsize=13, fontweight='bold', pad=15)
plt.xlabel('Week Number (1 to 145)', fontsize=11)
plt.ylabel('Total Orders (in Thousands)', fontsize=11)
plt.axhline(weekly_mean_orders / 1e3, color='gray', linestyle=':', label=f'Overall Mean ({weekly_mean_orders/1e3:.1f}K)')
plt.scatter([max_week], [weekly_max_orders / 1e3], color='red', s=80, zorder=5)
plt.annotate(f'Peak: Week {max_week}\n({weekly_max_orders/1e3:.1f}K orders)',
             xy=(max_week, weekly_max_orders / 1e3),
             xytext=(max_week + 4, weekly_max_orders / 1e3 - 50),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6))
plt.legend(frameon=True, facecolor='white', loc='lower right')
plt.tight_layout()
fig1_path = os.path.join(fig_dir, 'fig1_weekly_demand_trend.png')
plt.savefig(fig1_path)
plt.close()

# -------------------------------------------------------------
# Q2: Category and Meal Pareto (80/20)
# -------------------------------------------------------------
cat_summary = df.groupby('category').agg(
    total_orders=('num_orders', 'sum'),
    total_revenue=('revenue', 'sum')
).sort_values(by='total_orders', ascending=False).reset_index()
cat_summary['order_pct'] = (cat_summary['total_orders'] / overall_total_orders) * 100
cat_summary['cum_pct'] = cat_summary['order_pct'].cumsum()

meal_summary = df.groupby(['meal_id', 'category', 'cuisine']).agg(
    total_orders=('num_orders', 'sum'),
    mean_orders=('num_orders', 'mean'),
    std_orders=('num_orders', 'std'),
    total_revenue=('revenue', 'sum')
).sort_values(by='total_orders', ascending=False).reset_index()
meal_summary['order_pct'] = (meal_summary['total_orders'] / overall_total_orders) * 100
meal_summary['cum_pct'] = meal_summary['order_pct'].cumsum()
meal_summary['cv'] = meal_summary['std_orders'] / meal_summary['mean_orders']

top_80_cats = cat_summary[cat_summary['cum_pct'] <= 85]
top_80_meals = meal_summary[meal_summary['cum_pct'] <= 80]
print(f"Total categories: {len(cat_summary)}. Categories driving ~80% demand: {len(top_80_cats)}")
print(f"Top 4 categories share: {cat_summary.iloc[:4]['order_pct'].sum():.2f}%")
print(f"Total meals: {len(meal_summary)}. Meals driving 80% demand: {len(top_80_meals)} of 51 ({len(top_80_meals)/51*100:.1f}%)")

# Figure 2: Category Demand Pareto
plt.figure(figsize=(12, 6), dpi=300)
palette = ['#2ca02c' if i < 4 else '#a6d96a' if i < 7 else '#d9d9d9' for i in range(len(cat_summary))]
bars = plt.bar(cat_summary['category'], cat_summary['total_orders'] / 1e6, color=palette, edgecolor='black', linewidth=0.5)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.title('Beverages, Rice Bowls, and Sandwiches Generate Over 55% of Total Demand Across All Centers', fontsize=13, fontweight='bold', pad=15)
plt.xlabel('Meal Category', fontsize=11)
plt.ylabel('Total Orders (Millions)', fontsize=11)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.3, f'{yval:.1f}M', ha='center', va='bottom', fontsize=8.5)
plt.tight_layout()
fig2_path = os.path.join(fig_dir, 'fig2_category_demand_distribution.png')
plt.savefig(fig2_path)
plt.close()

# -------------------------------------------------------------
# Q3: Center Performance & Center Type
# -------------------------------------------------------------
center_perf = df.groupby(['center_id', 'center_type', 'city_code', 'op_area']).agg(
    total_orders=('num_orders', 'sum'),
    total_revenue=('revenue', 'sum')
).reset_index()

center_type_stats = center_perf.groupby('center_type').agg(
    center_count=('center_id', 'count'),
    mean_orders=('total_orders', 'mean'),
    median_orders=('total_orders', 'median'),
    total_orders=('total_orders', 'sum'),
    mean_op_area=('op_area', 'mean')
).reset_index()
center_type_stats['orders_share_pct'] = (center_type_stats['total_orders'] / overall_total_orders) * 100

print("Center Type Breakdown:")
print(center_type_stats)

# -------------------------------------------------------------
# Q4: Statistical Testing of Promotions (Mann-Whitney U)
# -------------------------------------------------------------
# Emailer
orders_email_1 = df[df['emailer_for_promotion'] == 1]['num_orders']
orders_email_0 = df[df['emailer_for_promotion'] == 0]['num_orders']
u_email, p_email = stats.mannwhitneyu(orders_email_1, orders_email_0, alternative='greater')
mean_email_1, median_email_1 = orders_email_1.mean(), orders_email_1.median()
mean_email_0, median_email_0 = orders_email_0.mean(), orders_email_0.median()
email_lift_mean = ((mean_email_1 - mean_email_0) / mean_email_0) * 100
email_lift_median = ((median_email_1 - median_email_0) / median_email_0) * 100

# Homepage Featured
orders_home_1 = df[df['homepage_featured'] == 1]['num_orders']
orders_home_0 = df[df['homepage_featured'] == 0]['num_orders']
u_home, p_home = stats.mannwhitneyu(orders_home_1, orders_home_0, alternative='greater')
mean_home_1, median_home_1 = orders_home_1.mean(), orders_home_1.median()
mean_home_0, median_home_0 = orders_home_0.mean(), orders_home_0.median()
home_lift_mean = ((mean_home_1 - mean_home_0) / mean_home_0) * 100
home_lift_median = ((median_home_1 - median_home_0) / median_home_0) * 100

# Both Combined
orders_both_1 = df[(df['emailer_for_promotion'] == 1) & (df['homepage_featured'] == 1)]['num_orders']
orders_neither = df[(df['emailer_for_promotion'] == 0) & (df['homepage_featured'] == 0)]['num_orders']
u_both, p_both = stats.mannwhitneyu(orders_both_1, orders_neither, alternative='greater')
both_lift_mean = ((orders_both_1.mean() - orders_neither.mean()) / orders_neither.mean()) * 100
both_lift_median = ((orders_both_1.median() - orders_neither.median()) / orders_neither.median()) * 100

print(f"Emailer Promo: Mean Lift = +{email_lift_mean:.1f}%, Median Lift = +{email_lift_median:.1f}%, p-val = {p_email}")
print(f"Homepage Featured: Mean Lift = +{home_lift_mean:.1f}%, Median Lift = +{home_lift_median:.1f}%, p-val = {p_home}")
print(f"Both Combined: Mean Lift = +{both_lift_mean:.1f}%, Median Lift = +{both_lift_median:.1f}%, p-val = {p_both}")

# Figure 3: Boxplot of Promotions Impact
df['promo_group'] = 'No Promotion'
df.loc[(df['emailer_for_promotion'] == 1) & (df['homepage_featured'] == 0), 'promo_group'] = 'Email Only'
df.loc[(df['emailer_for_promotion'] == 0) & (df['homepage_featured'] == 1), 'promo_group'] = 'Homepage Only'
df.loc[(df['emailer_for_promotion'] == 1) & (df['homepage_featured'] == 1), 'promo_group'] = 'Email + Homepage'

plt.figure(figsize=(10, 6), dpi=300)
order_groups = ['No Promotion', 'Email Only', 'Homepage Only', 'Email + Homepage']
sns.boxplot(
    data=df, 
    x='promo_group', 
    y='num_orders', 
    order=order_groups,
    palette=['#95a5a6', '#3498db', '#e67e22', '#e74c3c'],
    showfliers=False,
    boxprops=dict(alpha=0.85)
)
medians = [df[df['promo_group'] == g]['num_orders'].median() for g in order_groups]
means = [df[df['promo_group'] == g]['num_orders'].mean() for g in order_groups]
for i, (med, mn) in enumerate(zip(medians, means)):
    plt.text(i, med + 15, f'Median: {int(med)}\nMean: {int(mn)}', ha='center', va='bottom', fontweight='bold', fontsize=9, bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

plt.title('Promotions Significantly Multiply Orders: Combined Campaigns Yield Over 3.7x Higher Mean Demand (p < 0.001)', fontsize=12, fontweight='bold', pad=15)
plt.xlabel('Marketing Channel Combination', fontsize=11)
plt.ylabel('Number of Orders (Boxplot without extreme outliers)', fontsize=11)
plt.tight_layout()
fig3_path = os.path.join(fig_dir, 'fig3_promotions_impact_boxplot.png')
plt.savefig(fig3_path)
plt.close()

# -------------------------------------------------------------
# Q5: Discount vs. Demand & Spearman Correlation
# -------------------------------------------------------------
spearman_corr, spearman_p = stats.spearmanr(df['discount_rate'], df['num_orders'])
pearson_corr, pearson_p = stats.pearsonr(df['discount_rate'], df['num_orders'])
# Discount with/without promotions
avg_discount_no_promo = df[df['has_promo'] == 0]['discount_rate'].mean() * 100
avg_discount_with_promo = df[df['has_promo'] == 1]['discount_rate'].mean() * 100

print(f"Spearman Correlation (Discount vs. Orders): {spearman_corr:.4f} (p-val: {spearman_p})")
print(f"Pearson Correlation: {pearson_corr:.4f}")
print(f"Average Discount without promo: {avg_discount_no_promo:.2f}%, with promo: {avg_discount_with_promo:.2f}%")

# Figure 4: Discount vs. Orders Scatter / Binned Plot
# To avoid overplotting 456k points, we create a binned discount curve with sample scatter
df['discount_bin'] = pd.cut(df['discount_rate'] * 100, bins=[-1, 0, 5, 10, 15, 20, 30, 50, 100], 
                            labels=['0%', '1-5%', '6-10%', '11-15%', '16-20%', '21-30%', '31-50%', '>50%'])
bin_stats = df.groupby('discount_bin', observed=False).agg(
    mean_orders=('num_orders', 'mean'),
    median_orders=('num_orders', 'median'),
    count=('num_orders', 'count')
).reset_index()

plt.figure(figsize=(10, 6), dpi=300)
# Subsample for background density
sample = df.sample(n=3000, random_state=42)
plt.scatter(sample['discount_rate'] * 100, sample['num_orders'], alpha=0.15, color='#34495e', s=12, label='Sample Orders (n=3000)')
# Line for binned average
bin_centers = [0, 2.5, 8, 13, 18, 25.5, 40, 60]
plt.plot(bin_centers, bin_stats['mean_orders'], color='#e74c3c', lw=3, marker='o', label='Average Demand by Discount Bracket')
plt.title('Moderate Spearman Correlation (rs = 0.283): Demand Accelerates Sharply Above 15% Discount Threshold', fontsize=12, fontweight='bold', pad=15)
plt.xlabel('Promotional Discount Percentage (%)', fontsize=11)
plt.ylabel('Number of Orders', fontsize=11)
plt.ylim(0, 3500)
plt.xlim(-2, 75)
plt.legend(frameon=True, facecolor='white', loc='upper left')
plt.tight_layout()
fig4_path = os.path.join(fig_dir, 'fig4_discount_vs_demand_relationship.png')
plt.savefig(fig4_path)
plt.close()

# -------------------------------------------------------------
# Q6: Volatility (CV) and Safety Stock Ranking
# -------------------------------------------------------------
meal_volatility = df.groupby(['meal_id', 'category', 'cuisine']).agg(
    total_orders=('num_orders', 'sum'),
    mean_orders=('num_orders', 'mean'),
    std_orders=('num_orders', 'std'),
    min_orders=('num_orders', 'min'),
    max_orders=('num_orders', 'max')
).reset_index()

meal_volatility['cv'] = meal_volatility['std_orders'] / meal_volatility['mean_orders']
meal_volatility['volatility_rank'] = meal_volatility['cv'].rank(ascending=False)
meal_volatility['volume_rank'] = meal_volatility['total_orders'].rank(ascending=False)
meal_volatility['overall_risk_score'] = (meal_volatility['volatility_rank'] + meal_volatility['volume_rank']) / 2
meal_volatility = meal_volatility.sort_values(by='cv', ascending=False)

print("\nTop 5 Most Volatile Meals (Highest CV):")
print(meal_volatility[['meal_id', 'category', 'cuisine', 'total_orders', 'mean_orders', 'cv']].head())

print("\nTop 5 Most Stable Meals (Lowest CV):")
print(meal_volatility[['meal_id', 'category', 'cuisine', 'total_orders', 'mean_orders', 'cv']].tail())

# Save metrics CSV for notebook & report referencing
metrics_dir = os.path.join(base_dir, 'docs')
meal_volatility.to_csv(os.path.join(metrics_dir, 'meal_volatility_rankings.csv'), index=False)

# -------------------------------------------------------------
# Machine Learning Bonus: Demand Forecasting
# -------------------------------------------------------------
print("\n=== Training Machine Learning Models for Demand Forecasting ===")
# Feature preparation on train
df_ml = df.sort_values(by='week').copy()
# Features
feature_cols = ['checkout_price', 'base_price', 'discount_rate', 'emailer_for_promotion', 
                'homepage_featured', 'op_area', 'center_type', 'category', 'cuisine']
df_ml = pd.get_dummies(df_ml, columns=['center_type', 'category', 'cuisine'], drop_first=True)
encoded_feature_cols = [c for c in df_ml.columns if c in feature_cols or any(c.startswith(p) for p in ['center_type_', 'category_', 'cuisine_'])]

# Split: Last 10 weeks (weeks 136 to 145) as Validation/Holdout
train_split = df_ml[df_ml['week'] <= 135]
val_split = df_ml[df_ml['week'] >= 136]

X_train = train_split[encoded_feature_cols]
y_train = np.log1p(train_split['num_orders'])
X_val = val_split[encoded_feature_cols]
y_val = val_split['num_orders']

# Baseline 1: Historical Mean per center-meal
train_hist_mean = train_split.groupby(['center_id', 'meal_id'])['num_orders'].mean().reset_index().rename(columns={'num_orders': 'hist_mean'})
val_with_mean = val_split.merge(train_hist_mean, on=['center_id', 'meal_id'], how='left')
val_with_mean['hist_mean'] = val_with_mean['hist_mean'].fillna(train_split['num_orders'].mean())
mae_baseline_mean = mean_absolute_error(y_val, val_with_mean['hist_mean'])

# Baseline 2: Previous week order (Naive lag-1)
# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
pred_lr_log = lr.predict(X_val)
pred_lr = np.expm1(pred_lr_log)
pred_lr = np.maximum(0, pred_lr)
mae_lr = mean_absolute_error(y_val, pred_lr)

# Random Forest Regressor (Fast optimized parameters)
rf = RandomForestRegressor(n_estimators=50, max_depth=16, min_samples_leaf=10, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
pred_rf_log = rf.predict(X_val)
pred_rf = np.expm1(pred_rf_log)
pred_rf = np.maximum(0, pred_rf)
mae_rf = mean_absolute_error(y_val, pred_rf)

print(f"MAE Baseline (Historical Mean): {mae_baseline_mean:.2f}")
print(f"MAE Linear Regression: {mae_lr:.2f}")
print(f"MAE Random Forest: {mae_rf:.2f}")
print(f"Random Forest MAE Improvement vs Baseline: {((mae_baseline_mean - mae_rf) / mae_baseline_mean) * 100:.2f}%")

# Generate predictions on test.csv (for bonus submission)
test = pd.read_csv(os.path.join(raw_dir, 'test.csv'))
test_merged = test.merge(centers, on='center_id', how='left').merge(meals, on='meal_id', how='left')
test_merged['discount_rate'] = np.maximum(0, (test_merged['base_price'] - test_merged['checkout_price']) / test_merged['base_price'])
test_ml = pd.get_dummies(test_merged, columns=['center_type', 'category', 'cuisine'], drop_first=True)
# Align columns with X_train
for col in encoded_feature_cols:
    if col not in test_ml.columns:
        test_ml[col] = 0
X_test = test_ml[encoded_feature_cols]
test_pred_log = rf.predict(X_test)
test['num_orders'] = np.round(np.maximum(0, np.expm1(test_pred_log))).astype(int)
test[['id', 'num_orders']].to_csv(os.path.join(base_dir, 'reports', 'test_predictions.csv'), index=False)
print("Saved test_predictions.csv successfully!")
