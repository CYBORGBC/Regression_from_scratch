import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples
n_samples = 1000

# Feature generation
age = np.random.randint(18, 65, size=n_samples)                       # Age between 18 and 65
education_years = np.random.randint(0, 20, size=n_samples)            # 0 to 20 years of education
work_experience = np.maximum(age - 18, 0) * np.random.uniform(0.4, 0.9, size=n_samples) # Related to age
work_experience = work_experience.astype(int)

# Annual income (linear relation + some noise)
annual_income = (
    2000 * education_years + 
    500 * work_experience + 
    300 * age + 
    np.random.normal(0, 10000, size=n_samples)  # Some noise
)

# Ensure no negative income
annual_income = np.maximum(annual_income, 5000)

# Probability of owning a car depends on income, age, and work experience
logit = (
    0.00005 * annual_income + 
    0.02 * work_experience + 
    0.01 * age - 
    3  # bias
)

prob_owns_car = 1 / (1 + np.exp(-logit))  # Sigmoid
owns_car = np.random.binomial(1, prob_owns_car)

# Create DataFrame
df = pd.DataFrame({
    'Age': age,
    'Years_of_Education': education_years,
    'Work_Experience': work_experience,
    'Annual_Income': np.round(annual_income, 2),
    'Owns_Car': owns_car
})

# Show first few rows
print(df.head())

# Save to CSV if needed
df.to_csv('generated_dataset.csv', index=False)
