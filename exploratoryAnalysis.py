'''
John Guerrerio
Dali 2024 Winter Application
Last updated 2/3/2024

Exploratory analysis of the customers dataset to get a better feel for the data and explore directions for ML
'''

import pandas as pd

df = pd.read_csv('Superstore.csv')
pd.set_option('display.expand_frame_repr', False)  # turns off truncation when printing the dataset

print("Number of rows: " + str(df.shape[0]))
print()

print("Headings: ")
print(df.columns.values.tolist())
print()

print("Sample rows: ")
print(df.head())
print()

# looking at the sample rows, I see a lot of NaNs.  Let's figure out how many there are - this will influence how we deal with them
print("Order ID: " + str(df.isnull().sum()))
print()
# there's an equal number of missing values per column - I'm guessing these were added in randomly to make the challenge harder
# I assume the missing data is Missing Completely at Random
# In this case, dropping rows with missing data for columns of interest will not introduce bias
# However, I'd like to avoid doing so if possible
# For some quantitative features, we can replace missing values with the median for the column (e.g., Profit)
# For other features (e.g., Customer Name) we can't make an educated guess
# If these features are important for a model we develop later, we might have to drop the row

# Now let's generate some descriptive statistics for the quantitative columns
print("Quantity: ")
print("- Mean: " + str(df["Quantity"].mean()))
print("- Median: " + str(df["Quantity"].median()))
print("- Standard Deviation: " + str(df["Quantity"].std()))

print("Discount:")
print("- Mean: " + str(df["Discount"].mean()))
print("- Median: " + str(df["Discount"].median()))
print("- Standard Deviation: " + str(df["Discount"].std()))

print("Profit:")
print("- Mean: " + str(df["Profit"].mean()))
print("- Median: " + str(df["Profit"].median()))
print("- Standard Deviation: " + str(df["Profit"].std()))
print()

# Now let's generate some statistics for the qualitative columns:
print(df["Ship Mode"].value_counts())
print()

print("Number of unique customers: " + str(df["Customer ID"].nunique()))

onePurchase = 0
multiplePurcahses = 0
purchaseCounts = df["Customer ID"].value_counts()

for (customer, count) in purchaseCounts.items():
    if count == 1:
        onePurchase += 1
    else:
        multiplePurcahses += 1

print("One-time customers: " + str(onePurchase))
print("Repeat customers: " + str(multiplePurcahses))
print("Mean number of purchases per customer: " + str(purchaseCounts.mean()))
print("Median number of purchases per customer: " + str(purchaseCounts.median()))
print("Standard deviation number of purchases per customer: " + str(purchaseCounts.std()))
print()

print(df["Segment"].value_counts())
print()

print(df["Country"].value_counts())
print()
# only United States is interesting - I would assume a superstore would have some international sales
# in any case, country is not a useful feature

cities = df["City"].value_counts()
print(cities.head())  # cities with the top number of purchases
print("Number of unique cities: " + str(len(cities)))
print()

states = df["State"].value_counts()
print(states.head())  # states with the top number of purchases
print("Number of unique states: " + str(len(states)))  # not all 50 states made purchases
print()

regions = df["Region"].value_counts()
print(regions.head())
print()

category = df["Category"].value_counts()
print(category)
print()
# only three categories is surprising

subCategory = df["Sub-Category"].value_counts()
print(subCategory)
print()

# looking at this exploratory analysis, some potential questions that jump out at me for an ML model to predict are:
# - will a customer make more than or less than the number of median purchases per customer
# - will a purchase make above or below the median profit
# - can we cluster customers to recommend sub-categories of products a customer might be interested in?

# the next step of my analysis is to perform statistical tests to determine if there is a correlation between
# certain variables in this dataset.  This will inform feature selection and give me a deeper insight into the dataset
# please see the file correlation.py for these tests
