# Complete Python Pandas Project - Olympics & Coffee Sales Analysis

## Project Overview
This comprehensive pandas project covers all major concepts from the Complete Python Pandas Tutorial (2025 Updated Edition). The project uses two main datasets: Olympic Athletes data and Coffee Sales data.

---

## Table of Contents
1. Setting up Environment & Creating DataFrames
2. Accessing Data (.head, .tail, .sample, .loc, .iloc)
3. Setting Values in DataFrames
4. Sorting Data
5. Filtering Data (Numeric, Multiple Conditions, String Operations)
6. Adding & Removing Columns
7. Renaming Columns
8. Working with DateTime
9. Applying Custom Functions (Lambda)
10. Merging & Concatenating
11. Handling Missing Values
12. Aggregating Data (value_counts, GroupBy)
13. Pivot Tables
14. Advanced Operations (shift, rank, cumsum, rolling)
15. Statistical Analysis
16. Saving Data

---

## Dataset Schema

### Coffee Sales Dataset
- **day**: Day of the week (Monday-Sunday)
- **coffee_type**: Type of coffee (Latte, Espresso, Cappuccino, Mocha, Americano)
- **units_sold**: Number of units sold
- **revenue**: Revenue in USD

### Olympic Athletes Dataset
- **athlete_id**: Unique athlete identifier
- **name**: Athlete name
- **height_cm**: Height in centimeters
- **weight_kg**: Weight in kilograms
- **sport**: Type of sport
- **born_country**: Birth country code
- **born_city**: Birth city
- **born_date**: Birth date
- **medals**: Number of medals won

### Olympic Results Dataset
- **athlete_id**: Reference to athlete
- **event**: Olympic event name
- **medal**: Medal type (Gold, Silver, Bronze, None)
- **year**: Olympic year

---

## Key Concepts & Code Examples

### 1. Creating DataFrames

```python
import pandas as pd
import numpy as np

# Create from dictionary
df_coffee = pd.DataFrame({
    'day': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
    'coffee_type': ['Latte', 'Espresso', 'Cappuccino', 'Latte', 'Espresso', 'Cappuccino', 'Latte'],
    'units_sold': [25, 18, 22, 30, 15, 35, 28],
    'revenue': [124.75, 71.82, 99.0, 149.7, 59.85, 157.5, 139.72]
})

# Load from file
df_athletes = pd.read_csv('athletes.csv')
df_results = pd.read_parquet('results.parquet')
df_excel = pd.read_excel('data.xlsx', sheet_name='Olympics')
```

### 2. Accessing Data

```python
# View first/last/random rows
df.head()      # First 5 rows
df.head(3)     # First 3 rows
df.tail(2)     # Last 2 rows
df.sample(n=2) # 2 random rows

# Access by label (.loc)
df.loc[0, 'day']                    # Single value
df.loc[0:2, ['day', 'coffee_type']] # Rows 0-2, specific columns

# Access by position (.iloc)
df.iloc[0, 0]        # First row, first column
df.iloc[0:2, 0:2]    # Rows 0-2, columns 0-2

# Access columns
df['coffee_type']
df.coffee_type  # Only if no spaces in name
```

### 3. Setting Values

```python
# Set single value
df.loc[0, 'revenue'] = 150.0

# Set multiple values
df.loc[0:2, 'revenue'] = 100.0

# Set with condition
df.loc[df['coffee_type'] == 'Latte', 'revenue'] = 120.0
```

### 4. Sorting

```python
# Sort ascending
df.sort_values('units_sold')

# Sort descending
df.sort_values('units_sold', ascending=False)

# Sort by multiple columns
df.sort_values(['units_sold', 'coffee_type'], ascending=[False, True])
```

### 5. Filtering Data

#### Numeric Conditions
```python
# Single condition
tall_athletes = df_athletes[df_athletes['height_cm'] > 190]

# Multiple conditions
result = df_athletes[(df_athletes['height_cm'] > 190) & 
                     (df_athletes['born_country'] == 'USA')]

# OR condition
result = df_athletes[(df_athletes['medals'] >= 3) | 
                     (df_athletes['sport'] == 'Basketball')]
```

#### String Operations
```python
# Contains
df[df['name'].str.contains('Keith')]

# Case insensitive
df[df['name'].str.contains('Keith', case=False)]

# Regex pattern
df[df['name'].str.contains(r'^[A-M]')]  # Names starting with A-M

# String methods
df[df['born_city'].str.startswith('B')]
df[df['sport'].str.len() == 10]
df[df['name'].str.upper() == df['name'].str.upper()]
```

#### Query Method
```python
df.query('height_cm > 190 and sport == "Basketball"')
df.query('born_country == "USA"')
```

### 6. Adding Columns

```python
# Simple column addition
df['new_column'] = 100

# Based on condition (np.where)
df['price'] = np.where(
    df['coffee_type'] == 'Espresso',
    3.99,
    4.99
)

# From calculation
df['profit'] = df['revenue'] - (df['units_sold'] * 2)

# Apply lambda function
df['height_category'] = df['height_cm'].apply(
    lambda x: 'Short' if x < 170 else ('Average' if x < 190 else 'Tall')
)

# Apply custom function
def categorize_athlete(row):
    if row['height_cm'] < 175 and row['weight_kg'] < 70:
        return 'Light'
    elif row['height_cm'] < 185 and row['weight_kg'] < 80:
        return 'Medium'
    else:
        return 'Heavy'

df['category'] = df.apply(categorize_athlete, axis=1)
```

### 7. Removing Columns

```python
# Drop by column name
df = df.drop(columns=['column_name'])

# Drop by index
df = df.drop(0)  # Drop first row

# Keep only specific columns
df = df[['col1', 'col2', 'col3']]

# Drop with inplace
df.drop(columns=['column_name'], inplace=True)
```

### 8. Renaming Columns

```python
# Rename single column
df = df.rename(columns={'old_name': 'new_name'})

# Rename multiple columns
df = df.rename(columns={
    'height_cm': 'height_centimeters',
    'weight_kg': 'weight_kilograms'
})
```

### 9. Working with DateTime

```python
# Convert to datetime
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

# Extract components
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['day_name'] = df['date'].dt.day_name()
df['week'] = df['date'].dt.isocalendar().week

# Date operations
df['is_leap_year'] = df['date'].dt.is_leap_year
df['quarter'] = df['date'].dt.quarter
df['days_since'] = (pd.Timestamp.now() - df['date']).dt.days
```

### 10. Merging DataFrames

```python
# Inner merge
result = pd.merge(df1, df2, on='key', how='inner')

# Left merge
result = pd.merge(df1, df2, on='key', how='left')

# Right merge
result = pd.merge(df1, df2, on='key', how='right')

# Outer merge
result = pd.merge(df1, df2, on='key', how='outer')

# Merge on multiple keys
result = pd.merge(df1, df2, on=['key1', 'key2'])
```

### 11. Concatenating DataFrames

```python
# Concatenate vertically (stack)
result = pd.concat([df1, df2], ignore_index=True)

# Concatenate horizontally (side by side)
result = pd.concat([df1, df2], axis=1)

# Concatenate with keys
result = pd.concat([df1, df2], keys=['first', 'second'])
```

### 12. Handling Missing Values

```python
# Check for null values
df.isnull()
df.isnull().sum()

# Drop null values
df.dropna()                    # Drop any row with NaN
df.dropna(subset=['column1'])  # Drop if specific column is NaN
df.dropna(how='all')           # Drop only all-NaN rows

# Fill null values
df['column'].fillna(0)                    # Fill with constant
df['column'].fillna(df['column'].mean())  # Fill with mean
df['column'].fillna(method='ffill')       # Forward fill
df['column'].fillna(method='bfill')       # Backward fill

# Interpolate
df['column'].interpolate()     # Linear interpolation
df['column'].interpolate(method='polynomial', order=2)
```

### 13. Aggregating Data

```python
# Value counts
df['column'].value_counts()
df['column'].value_counts().head(10)

# Sum
df['column'].sum()

# Mean, Median, Std
df['column'].mean()
df['column'].median()
df['column'].std()

# Min, Max
df['column'].min()
df['column'].max()
df['column'].nlargest(5)
df['column'].nsmallest(5)
```

### 14. GroupBy Operations

```python
# Single column groupby
df.groupby('category')['revenue'].sum()
df.groupby('category')['revenue'].mean()
df.groupby('category').size()

# Multiple columns groupby
df.groupby(['category', 'region'])['revenue'].sum()

# Multiple aggregations
df.groupby('sport').agg({
    'height_cm': ['mean', 'min', 'max'],
    'weight_kg': 'mean',
    'medals': 'sum'
})

# Named aggregations
df.groupby('sport').agg(
    avg_height=('height_cm', 'mean'),
    total_medals=('medals', 'sum'),
    athlete_count=('athlete_id', 'count')
)
```

### 15. Pivot Tables

```python
# Basic pivot
pivot = df.pivot_table(
    values='revenue',
    index='month',
    columns='coffee_type',
    aggfunc='sum'
)

# Multiple aggregations
pivot = df.pivot_table(
    values=['revenue', 'units'],
    index='month',
    columns='coffee_type',
    aggfunc=['sum', 'mean']
)

# Fill missing values in pivot
pivot.fillna(0)
```

### 16. Advanced Operations

```python
# Shift (lag/lead)
df['prev_value'] = df['column'].shift(1)
df['next_value'] = df['column'].shift(-1)

# Rank
df['rank'] = df['column'].rank()
df['rank_desc'] = df['column'].rank(ascending=False)

# Cumulative sum
df['cumsum'] = df['column'].cumsum()

# Cumulative product
df['cumprod'] = df['column'].cumprod()

# Difference from previous
df['diff'] = df['column'].diff()

# Percentage change
df['pct_change'] = df['column'].pct_change() * 100

# Rolling mean (moving average)
df['rolling_mean'] = df['column'].rolling(window=3).mean()

# Rolling sum
df['rolling_sum'] = df['column'].rolling(window=3).sum()

# Expanding operations
df['expanding_mean'] = df['column'].expanding().mean()
```

### 17. Saving Data

```python
# Save to CSV
df.to_csv('filename.csv', index=False)

# Save to Excel
df.to_excel('filename.xlsx', sheet_name='Sheet1', index=False)

# Save to Parquet (efficient format)
df.to_parquet('filename.parquet')

# Save to JSON
df.to_json('filename.json')

# Save to SQL
df.to_sql('table_name', connection, if_exists='replace')
```

---

## Complete Example Workflow

```python
import pandas as pd
import numpy as np

# 1. Load data
df_athletes = pd.read_csv('athletes.csv')
df_results = pd.read_csv('results.csv')

# 2. Explore data
print(df_athletes.head())
print(df_athletes.info())
print(df_athletes.describe())

# 3. Clean data - handle missing values
df_athletes['height_cm'].fillna(df_athletes['height_cm'].mean(), inplace=True)
df_athletes.dropna(subset=['weight_kg'], inplace=True)

# 4. Add new columns
df_athletes['height_category'] = df_athletes['height_cm'].apply(
    lambda x: 'Tall' if x > 190 else 'Average' if x > 170 else 'Short'
)

# 5. Filter data
tall_usa_athletes = df_athletes[
    (df_athletes['height_cm'] > 190) & 
    (df_athletes['born_country'] == 'USA')
]

# 6. Merge datasets
df_athlete_results = pd.merge(df_athletes, df_results, on='athlete_id')

# 7. Aggregate data
medals_by_sport = df_athletes.groupby('sport')['medals'].agg(['sum', 'mean', 'count'])

# 8. Create pivot table
results_pivot = df_results.pivot_table(
    values='medal',
    index='sport',
    columns='year',
    aggfunc='count',
    fill_value=0
)

# 9. Advanced analysis
athlete_ranking = df_athletes.groupby('sport').agg({
    'medals': 'sum',
    'height_cm': 'mean',
    'weight_kg': 'mean'
}).sort_values('medals', ascending=False)

# 10. Save results
athlete_ranking.to_csv('athlete_analysis.csv')
tall_usa_athletes.to_excel('tall_usa_athletes.xlsx', index=False)
```

---

## Best Practices

1. **Use vectorized operations** instead of loops for better performance
2. **Handle missing data** explicitly (dropna, fillna, interpolate)
3. **Use groupby** for aggregate operations instead of loops
4. **Leverage column operations** (df['col1'] * df['col2'])
5. **Use .copy()** when you want to avoid SettingWithCopyWarning
6. **Set inplace=True** when you want to modify original DataFrame
7. **Use .loc[] for labels** and .iloc[] for position-based access
8. **Create meaningful column names** for better readability
9. **Use merge/concat** for combining DataFrames instead of loops
10. **Profile your code** to identify bottlenecks

---

## Common Errors & Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| SettingWithCopyWarning | Modifying copied/filtered DataFrame | Use .copy() or .loc[] for assignment |
| KeyError | Column name doesn't exist | Check df.columns for correct name |
| ValueError: cannot reindex | Index/columns mismatch in merge | Specify correct 'on' parameter in merge |
| NaN in results | Missing values in data | Use fillna(), dropna(), or interpolate() |
| Memory error | DataFrame too large | Use chunking, Parquet format, or filters |

---

## Performance Tips

1. Use **Parquet** format for large files (faster than CSV)
2. Specify **dtypes** when loading data to reduce memory
3. Use **categorical** dtype for repeated string values
4. Filter data early in your pipeline
5. Use **.astype()** to convert types for memory efficiency
6. Avoid iterating rows - use vectorized operations
7. Use **query()** method for complex filtering
8. Leverage **copy-on-write** (COW) behavior in pandas 2.0+

---

## Pandas 2.0 New Features

1. **PyArrow backend** for better performance and memory efficiency
2. **Improved string operations** with better performance
3. **Copy-on-write** mode by default
4. **New data types** support
5. **Performance improvements** in groupby and merge operations
6. **Better integration** with numpy and other libraries

---

## Resources

- **GitHub Repo**: https://github.com/KeithGalli/complete-pandas-tutorial
- **Pandas Documentation**: https://pandas.pydata.org/docs/
- **Pandas 2.0 Blog**: https://datapythonista.me/blog/pandas-20-and-the-arrow-revolution-part-i
- **StrataScratch**: Practice pandas with real problems
- **100 Pandas Problems**: Challenge yourself with curated problems

---

## Next Steps

1. Practice with your own datasets
2. Explore advanced topics (MultiIndex, Time Series, Window Functions)
3. Learn SQL for database operations
4. Study data visualization (Matplotlib, Seaborn, Plotly)
5. Master statistical analysis with SciPy and Statsmodels
6. Explore machine learning with Scikit-learn

---

**Happy Learning! 🐼📊**
