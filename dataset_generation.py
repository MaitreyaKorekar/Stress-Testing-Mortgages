import pandas as pd
import numpy as np
import random

np.random.seed(42)
random.seed(42)

n = 10000  # Number of synthetic records

# Helper function
def generate_categorical_distribution(choices, probabilities, size):
    probabilities = np.array(probabilities)
    probabilities = probabilities / probabilities.sum()
    return np.random.choice(choices, size=size, p=probabilities)

# Borrower ID
borrower_id = [f"B{str(i).zfill(5)}" for i in range(1, n + 1)]

# Gender
genders = generate_categorical_distribution(['Male', 'Female'], [0.487, 0.513], n)

# Age
age_bins = {
    '18-24': (18, 24, 0.061),
    '25-34': (25, 34, 0.212),
    '35-54': (35, 54, 0.531),
    '55+': (55, 75, 0.196)
}
ages = []
for label, (low, high, pct) in age_bins.items():
    count = int(pct * n)
    ages.extend(np.random.randint(low, high + 1, count))
ages = np.array(ages)
if len(ages) < n:
    ages = np.append(ages, np.random.randint(35, 54, n - len(ages)))

# Family Information
marital_status = generate_categorical_distribution(["Single", "Married", "Divorced", "Widowed"],
                                                   [0.4, 0.45, 0.1, 0.05], n)
dependents = np.random.choice([0, 1, 2, 3, 4], size=n, p=[0.3, 0.25, 0.25, 0.15, 0.05])

# Immigration status
immigration = generate_categorical_distribution(
    ['Canadian-born', 'Long-term immigrant', 'Recent immigrant'],
    [0.794, 0.137, 0.069], n
)

# Education level based on age (age restriction)
def get_age_restricted_education_levels(age):
    if age == 18:
        return ["No certificate, diploma or degree", "High school diploma or equivalent", "Trades or apprenticeship certificate"]
    elif age == 21:
        return [
            "No certificate, diploma or degree", "High school diploma or equivalent", "Trades or apprenticeship certificate",
            "College or CEGEP certificate or diploma", "University transfer program", "University certificate or diploma below a bachelor's degree", "Bachelor's degree"
        ]
    elif age == 22:
        return [
            "No certificate, diploma or degree", "High school diploma or equivalent", "Trades or apprenticeship certificate",
            "College or CEGEP certificate or diploma", "University transfer program", "University certificate or diploma below a bachelor's degree", "Bachelor's degree",
            "University certificate or diploma above a bachelor's degree", "First professional degree", "Master's degree"
        ]
    else:  # age >= 26
        return [
            "No certificate, diploma or degree", "High school diploma or equivalent", "Trades or apprenticeship certificate",
            "College or CEGEP certificate or diploma", "University transfer program", "University certificate or diploma below a bachelor's degree", "Bachelor's degree",
            "University certificate or diploma above a bachelor's degree", "First professional degree", "Master's degree", "Doctoral degree"
        ]

# Education levels and their probabilities
education_levels = [
    "No certificate, diploma or degree", "High school diploma or equivalent",
    "Trades or apprenticeship certificate", "College or CEGEP certificate or diploma",
    "University transfer program", "University certificate or diploma below a bachelor's degree",
    "Bachelor's degree", "University certificate or diploma above a bachelor's degree",
    "First professional degree", "Master's degree", "Doctoral degree"
]
edu_probs = [0.028, 0.166, 0.090, 0.269, 0.002, 0.043, 0.233, 0.045, 0.022, 0.091, 0.011]

# Generate education level based on age and probabilities
education = []
for age in ages:
    valid_education_levels = get_age_restricted_education_levels(age)
    # Normalize probabilities to match the filtered valid education levels
    valid_probs = [edu_probs[education_levels.index(level)] for level in valid_education_levels]
    valid_probs = np.array(valid_probs) / sum(valid_probs)  # Normalize to sum to 1
    selected_education = np.random.choice(valid_education_levels, p=valid_probs)
    education.append(selected_education)

# Employment Details
job_titles = ['Manual Labor', 'Tech Professional', 'Healthcare', 'Finance', 'Supply Chain', 'Retail', 'Education']
job_title = np.random.choice(job_titles, size=n)
years_with_current_employer = np.random.geometric(p=0.2, size=n)
total_years_employment = years_with_current_employer + np.random.randint(0, 20, size=n)
income_type = np.random.choice(['Salaried', 'Hourly', 'Contract'], size=n, p=[0.7, 0.2, 0.1])
employment_status = generate_categorical_distribution(
    ["Employed", "Self-employed", "Unemployed", "Student", "Retired"],
    [0.65, 0.15, 0.1, 0.05, 0.05], n
)

# Industry Sector
industry_sector = generate_categorical_distribution([
    "Tech", "Manual Labor", "Finance", "Healthcare", "Education", "Retail", "Logistics", "Construction", "Manufacturing", "Other"
], [0.15, 0.15, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05], n)

# Income deciles
income_deciles = [
    ("Lowest", 24100, 16000),
    ("Second", 32000, 28300),
    ("Third", 39400, 35900),
    ("Fourth", 46200, 42700),
    ("Fifth", 53400, 49800),
    ("Sixth", 60700, 57100),
    ("Seventh", 69500, 65000),
    ("Eighth", 80700, 74800),
    ("Ninth", 99600, 89200),
    ("Highest", 160000, 137700)
]
income_probs = np.array([2.7, 4.7, 6.0, 7.2, 8.4, 9.6, 10.9, 12.5, 15.0, 23.1])
income_probs = income_probs / income_probs.sum()
decile_labels = [d[0] for d in income_deciles]
income_decile = generate_categorical_distribution(decile_labels, income_probs, n)
decile_to_income = {d[0]: d[2] for d in income_deciles}
income = np.array([np.random.normal(loc=decile_to_income[decile], scale=0.1 * decile_to_income[decile]) for decile in income_decile])
income = np.clip(income, a_min=5000, a_max=None)

# Wealth Indicators (scaled with income decile)
decile_to_asset_multiplier = {
    "Lowest": 1.2, "Second": 1.4, "Third": 1.6, "Fourth": 1.8, "Fifth": 2.0,
    "Sixth": 2.2, "Seventh": 2.4, "Eighth": 2.6, "Ninth": 2.8, "Highest": 3.0
}
base_asset = 100000
assets = np.array([np.random.normal(loc=base_asset * decile_to_asset_multiplier[decile], scale=20000)
                   for decile in income_decile])
assets = np.clip(np.round(assets, -3), a_min=5000, a_max=None)
liabilities_percent = np.random.uniform(0.2, 0.8, size=n)
total_liabilities = (assets * liabilities_percent).round(2)
net_worth = (assets - total_liabilities).round(2)
liquid_assets_percent = np.random.uniform(0.1, 0.5, size=n)
liquid_assets = (assets * liquid_assets_percent).round(2)

# Loan Info with logic-based dependencies
number_of_open_loans = np.random.poisson(2, size=n)
total_value_of_open_loans = np.where(number_of_open_loans == 0, 0,
                                     total_liabilities + np.random.normal(loc=10000, scale=5000, size=n))
delinquencies_past_12mo = np.where(number_of_open_loans == 0, 0,
                                   np.random.poisson(0.2, size=n))

# Credit Info
credit_score = np.random.normal(680, 50, n).astype(int)
credit_utilization = np.clip(np.random.beta(2, 5, size=n) * 100, 0, 100)

# Mortgage Details
property_value = np.random.normal(900000, 150000, n).astype(int)
loan_to_value = np.random.uniform(60, 95, size=n)
mortgage_amount = (property_value * loan_to_value / 100).astype(int)
monthly_payment = np.random.normal(loc=1600, scale=500, size=n).clip(min=200)
interest_rate = np.round(np.random.normal(5, 1.2, n), 2)
term = np.random.choice([15, 20, 25, 30], size=n, p=[0.1, 0.2, 0.4, 0.3])

# Mortgage Start Year Logic:
# Calculate the year when the borrower turns 18
turn_18_year = 2024 - (ages - 18)

# Ensure the mortgage start year is after the borrower turns 18, and it can't be in the future.
mortgage_start_year = [
    np.random.choice(np.arange(turn_18, 2024 + 1))  # Ensure mortgage start year is at least when they turn 18
    for turn_18 in turn_18_year
]

# Assemble DataFrame
df = pd.DataFrame({
    "borrower_id": borrower_id,
    'gender': genders,
    'age': ages,
    'education_level': education,
    'immigration_status': immigration,
    "marital_status": marital_status,
    "dependents": dependents,
    'job_title': job_title,
    "industry_sector": industry_sector,
    "employment_status": employment_status,
    'annual_income': np.round(income, 2),
    'income_decile': income_decile,
    'years_with_current_employer': years_with_current_employer,
    'total_years_employment': total_years_employment,
    'income_type': income_type,
    'assets': assets,
    "total_liabilities": total_liabilities.round(2),
    "net_worth": net_worth.round(2),
    "liquid_assets": liquid_assets.round(2),
    "number_of_open_loans": number_of_open_loans,
    "total_value_of_open_loans": np.round(total_value_of_open_loans, 2),
    "delinquencies_past_12mo": delinquencies_past_12mo,
    'credit_score': credit_score,
    'credit_utilization': np.round(credit_utilization, 2),
    'property_value': property_value,
    'mortgage_amount': mortgage_amount,
    'loan_to_value': np.round(loan_to_value, 2),
    "monthly_payment": np.round(monthly_payment, 2),
    'interest_rate': interest_rate,
    'term_years': term,
    "mortgage_start_year": mortgage_start_year
})

# Save CSV
df.to_csv("synthetic_mortgage_data.csv", index=False)
print("Synthetic dataset generated and saved as 'synthetic_mortgage_data.csv'")