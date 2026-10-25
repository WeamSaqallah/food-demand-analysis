import pandas as pd
import numpy as np

train = pd.read_csv('food-demand-analysis/data/raw/train.csv')
centers = pd.read_csv('food-demand-analysis/data/raw/fulfilment_center_info.csv')
meals = pd.read_csv('food-demand-analysis/data/raw/meal_info.csv')

print('=== 1. MERGE AUDIT ===')
train_rows = len(train)
train_sum_orders = train['num_orders'].sum()
print(f'Train rows: {train_rows:,}, Sum num_orders: {train_sum_orders:,}')

merged = train.merge(centers, on='center_id', how='left').merge(meals, on='meal_id', how='left')
merged_rows = len(merged)
merged_sum_orders = merged['num_orders'].sum()
print(f'Merged rows: {merged_rows:,}, Sum num_orders: {merged_sum_orders:,}')
assert train_rows == merged_rows, 'Rows count changed!'
assert train_sum_orders == merged_sum_orders, 'Sum of orders changed!'
print('>> Perfect Join! Row count and total orders preserved exactly.')

print('\n=== 2. MISSING VALUES ===')
null_counts = merged.isnull().sum()
print(null_counts)

print('\n=== 3. DUPLICATES ===')
full_dups = merged.duplicated().sum()
key_dups = merged.duplicated(subset=['week', 'center_id', 'meal_id']).sum()
print(f'Exact full duplicates: {full_dups}')
print(f'Duplicate (week, center_id, meal_id) combinations: {key_dups}')

print('\n=== 4. PRICING INCONSISTENCY (checkout_price > base_price) ===')
higher_price = merged[merged['checkout_price'] > merged['base_price']]
print(f'Cases where checkout_price > base_price: {len(higher_price):,} ({len(higher_price)/len(merged)*100:.3f}%)')
if len(higher_price) > 0:
    diff = higher_price['checkout_price'] - higher_price['base_price']
    print(f'Difference stats: min={diff.min():.2f}, mean={diff.mean():.2f}, max={diff.max():.2f}')
    print('Sample cases:')
    print(higher_price[['id', 'week', 'center_id', 'meal_id', 'checkout_price', 'base_price', 'emailer_for_promotion', 'homepage_featured', 'num_orders']].head())

print('\n=== 5. OUTLIERS IN num_orders ===')
q25 = merged['num_orders'].quantile(0.25)
q75 = merged['num_orders'].quantile(0.75)
iqr = q75 - q25
upper_bound_iqr = q75 + 1.5 * iqr
p99 = merged['num_orders'].quantile(0.99)
p999 = merged['num_orders'].quantile(0.999)
max_order = merged['num_orders'].max()
print(f'min: {merged["num_orders"].min()}, median: {merged["num_orders"].median()}, mean: {merged["num_orders"].mean():.2f}')
print(f'IQR upper bound (1.5 IQR): {upper_bound_iqr:.1f}, count > upper_iqr: {(merged["num_orders"] > upper_bound_iqr).sum():,}')
print(f'99th percentile: {p99}, 99.9th percentile: {p999}, max: {max_order}')
extreme_orders = merged[merged['num_orders'] > p999]
print(f'Cases above 99.9th percentile: {len(extreme_orders)}')
print('Sample of extreme orders:')
print(extreme_orders[['id', 'week', 'center_id', 'meal_id', 'emailer_for_promotion', 'homepage_featured', 'checkout_price', 'num_orders']].head())

print('\n=== 6. WEEKS AND MEAL-CENTER COVERAGE ===')
total_weeks = merged['week'].nunique()
print(f'Unique weeks: {total_weeks} (min={merged["week"].min()}, max={merged["week"].max()})')
center_meal_combos = merged.groupby(['center_id', 'meal_id'])['week'].nunique()
print(f'Total active center-meal pairs: {len(center_meal_combos):,}')
print(f'Active pairs with full 145 weeks: {(center_meal_combos == 145).sum():,}')
print(f'Active pairs with < 145 weeks: {(center_meal_combos < 145).sum():,}')
print(f'Distribution of weeks per pair:')
print(center_meal_combos.describe())
