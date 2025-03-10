# Capestone_project
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, zscore

# Load datasets
male_file = '/mnt/data/nhanes_adult_male_bmx_2020.csv'
female_file = '/mnt/data/nhanes_adult_female_bmx_2020.csv'

male_df = pd.read_csv(male_file)
female_df = pd.read_csv(female_file)

# Convert to numpy matrices
male = male_df.to_numpy()
female = female_df.to_numpy()

# Extract weight data for visualization
male_weights = male[:, 0]
female_weights = female[:, 0]

# Plot histograms
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 6), sharex=True)
axes[0].hist(female_weights, bins=20, alpha=0.7, color='red', edgecolor='black')
axes[0].set_title('Histogram of Female Weights')
axes[1].hist(male_weights, bins=20, alpha=0.7, color='blue', edgecolor='black')
axes[1].set_title('Histogram of Male Weights')
plt.xlim([min(male_weights.min(), female_weights.min()), max(male_weights.max(), female_weights.max())])
plt.xlabel('Weight (kg)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Boxplot comparison
plt.figure(figsize=(6, 4))
plt.boxplot([female_weights, male_weights], labels=['Female', 'Male'], patch_artist=True, widths=0.5)
plt.title('Boxplot of Male and Female Weights')
plt.ylabel('Weight (kg)')
plt.show()

# Compute statistics
stats = {
    "Male": {
        "Mean": np.mean(male_weights),
        "Median": np.median(male_weights),
        "Std Dev": np.std(male_weights),
        "Skewness": pd.Series(male_weights).skew()
    },
    "Female": {
        "Mean": np.mean(female_weights),
        "Median": np.median(female_weights),
        "Std Dev": np.std(female_weights),
        "Skewness": pd.Series(female_weights).skew()
    }
}
print(pd.DataFrame(stats))

# Add BMI column to female matrix
female_bmi = female[:, 0] / ((female[:, 1] / 100) ** 2)
female = np.column_stack((female, female_bmi))

# Standardize female matrix
zfemale = zscore(female, axis=0)

# Scatterplot matrix
sns.pairplot(pd.DataFrame(zfemale[:, [1, 0, 6, 5, 7]], columns=['Height', 'Weight', 'Waist Circ.', 'Hip Circ.', 'BMI']))
plt.show()

# Compute Pearson and Spearman correlations
correlations = {}
for i, var1 in enumerate(['Height', 'Weight', 'Waist Circ.', 'Hip Circ.', 'BMI']):
    for j, var2 in enumerate(['Height', 'Weight', 'Waist Circ.', 'Hip Circ.', 'BMI']):
        if i < j:
            pearson_corr, _ = pearsonr(zfemale[:, i], zfemale[:, j])
            spearman_corr, _ = spearmanr(zfemale[:, i], zfemale[:, j])
            correlations[f'{var1} - {var2}'] = {'Pearson': pearson_corr, 'Spearman': spearman_corr}

print(pd.DataFrame(correlations))

# Compute new ratios
female_waist_height = female[:, 6] / female[:, 1]
female_waist_hip = female[:, 6] / female[:, 5]
female = np.column_stack((female, female_waist_height, female_waist_hip))

male_waist_height = male[:, 6] / male[:, 1]
male_waist_hip = male[:, 6] / male[:, 5]
male = np.column_stack((male, male_waist_height, male_waist_hip))

# Boxplot comparison for new ratios
plt.figure(figsize=(8, 5))
plt.boxplot([female_waist_height, male_waist_height, female_waist_hip, male_waist_hip],
            labels=['Female WHtR', 'Male WHtR', 'Female WHR', 'Male WHR'], patch_artist=True, widths=0.5)
plt.title('Boxplot of Waist-to-Height and Waist-to-Hip Ratios')
plt.ylabel('Ratio')
plt.show()

# Extract lowest and highest BMI individuals
sorted_indices = np.argsort(female[:, 7])
lowest_bmi = zfemale[sorted_indices[:5]]
highest_bmi = zfemale[sorted_indices[-5:]]
print("Lowest BMI individuals:")
print(lowest_bmi)
print("Highest BMI individuals:")
print(highest_bmi)
