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

# pretty safe to drop as long as data isn't scarce
# numerical values use median

# take product name and predict category and subcategory
# vector distance

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
# - will a purchase make above or below the median profit
# - can we cluster customers to recommend sub-categories of products a customer might be interested in?

# the next step of my analysis is to perform statistical tests to determine if there is a correlation between
# certain variables in this dataset.  This will inform feature selection and give me a deeper insight into the dataset
# please see the file correlation.py for these tests


# above or below median profit good

# k means clustering - what types of customers you have

# semtantically what is similar to product name
# linear - given state and products, output would be a reccomended product

# predict what sub-category might be good, vast majority returning customers
# array of what each customer bought, drop an element from the array and try and predict it

# no performance benchmarks we need to be hitting

# see if these models work, why you chose the model you chose

# you understand what you're doing and have justifications for why you're implementing cetrain things

#######################
# - predict product category for name (linear regression, BERT embeddings, word2vec, etc) - FOCUS ON NLP SUB CHALLENGE
# - reccomend useful products (embeddings from what you already bought, what's close)
# - useful subcategories based on prior purchases (drop 1 purchase as prediciton, ground truth is all purchases) - hardest, save for last
# - predict if purchase is above or below median profit (linear regression for simplicity, neural network, quadratic or cubic function, logarithmic function, random forests?)
# - try and cluster, get more info about the customers
# - do quantiy and discount correlate
# - tests - useful correlations for a superstore
# pull out features with highest weights to determine what influences profit the most
# ways to make the operation more efficient
# fill in missing category from sub category and vice versa

# good to see cohesiveness
# put all of this together

# cohesiveness - things that are useful to a superstore

# not many people have tried NLP

# the optional challenges are a bonus
# once you have a good amount of work on this, shift to optional challenges
# don't worry about computer vision challenge

# code samples don't have to be super complicated
# is code neat, organized, readable

# reccomendations might be a cool thing to do
# all sounds good
# explain your rationale
# comments really important