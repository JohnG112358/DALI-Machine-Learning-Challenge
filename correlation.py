'''
John Guerrerio
Dali 2024 Winter Application
Last updated 2/3/2024

Exploratory analysis of the customers dataset to determine if there is a correlation between different features and a dependent variable
Useful for feature engineering when it comes time to develop ML models
'''

from scipy.stats import f_oneway
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np

# For all of these tests, we consider alpha < 0.05 to be significant

### Correlation between region and profit ###
# Independent variable: region (categorical variable)
# Dependent variable: profit (continuous variable)
# To determine if there is a correlation between regon and profit, we use an ANOVA test. We
# make the following assumptions:
# - The distribution of sales for each region is normal
# - The variance of the distributions of sales for each region are equal
# - Each sale is independent from the others

df = pd.read_csv('Superstore.csv')

# we ignore rows with null region and/or null profit - substituting in values could change the proeprties of the distribution
# remember, we assume entries are missing data at random

west = df.loc[df['Region'] == "West"]["Profit"].dropna().tolist()
south = df.loc[df['Region'] == "South"]["Profit"].dropna().tolist()
east = df.loc[df['Region'] == "Central"]["Profit"].dropna().tolist()
central = df.loc[df['Region'] == "Central"]["Profit"].dropna().tolist()

print(f_oneway(west, south, central, east))
# pvalue=0.0009460319609385047 - results are statistically significant


### Correlation between category and profit ###
# Independent variable: category (categorical variable)
# Dependent variable: profit (continuous variable)
# we use an ANOVA test with the same assumptions as before
# we ignore rows with null region and/or null profit - substituting in values could change the proeprties of the distribution
# remember, we assume entries are missing data at random

office = df.loc[df['Category'] == "Office Supplies"]["Profit"].dropna().tolist()
furniture = df.loc[df['Category'] == "Furniture"]["Profit"].dropna().tolist()
technology = df.loc[df['Category'] == "Technology"]["Profit"].dropna().tolist()
print(f_oneway(office, furniture, technology))
# pvalue = 2.1206374409101864e-21, statistically significant results

# now we repeat the same test for subcategory
binders = df.loc[df['Sub-Category'] == "Binders"]["Profit"].dropna().tolist()
paper = df.loc[df['Sub-Category'] == "Paper"]["Profit"].dropna().tolist()
furnishings = df.loc[df['Sub-Category'] == "Furnishings"]["Profit"].dropna().tolist()
phones = df.loc[df['Sub-Category'] == "Phones"]["Profit"].dropna().tolist()
storage = df.loc[df['Sub-Category'] == "Storage"]["Profit"].dropna().tolist()
art = df.loc[df['Sub-Category'] == "Art"]["Profit"].dropna().tolist()
accessories = df.loc[df['Sub-Category'] == "Accessories"]["Profit"].dropna().tolist()
chairs = df.loc[df['Sub-Category'] == "Chairs"]["Profit"].dropna().tolist()
appliances = df.loc[df['Sub-Category'] == "Appliances"]["Profit"].dropna().tolist()
labels = df.loc[df['Sub-Category'] == "Labels"]["Profit"].dropna().tolist()
tables = df.loc[df['Sub-Category'] == "Tables"]["Profit"].dropna().tolist()
envelopes = df.loc[df['Sub-Category'] == "Envelopes"]["Profit"].dropna().tolist()
bookcases = df.loc[df['Sub-Category'] == "Bookcases"]["Profit"].dropna().tolist()
fasteners = df.loc[df['Sub-Category'] == "Fasteners"]["Profit"].dropna().tolist()
supplies = df.loc[df['Sub-Category'] == "Supplies"]["Profit"].dropna().tolist()
machines = df.loc[df['Sub-Category'] == "Machines"]["Profit"].dropna().tolist()
copiers = df.loc[df['Sub-Category'] == "Copiers"]["Profit"].dropna().tolist()

print(f_oneway(binders, paper, furnishings, phones, storage, art, accessories, chairs, appliances, labels,
               tables, envelopes, bookcases, fasteners, supplies, machines, copiers))
# pvalue=5.206200411171667e-147 - statistically significant


### Correlation between shipping method and profit ###
# Same anova test with the same assumptions
standard = df.loc[df['Ship Mode'] == "Standard Class"]["Profit"].dropna().tolist()
second = df.loc[df['Ship Mode'] == "Second Class"]["Profit"].dropna().tolist()
first = df.loc[df['Ship Mode'] == "First Class"]["Profit"].dropna().tolist()
same = df.loc[df['Ship Mode'] == "Same Day"]["Profit"].dropna().tolist()

print(f_oneway(standard, second, first, same))
# pvalue=0.8265487674803774 - not statistically significant




### Correlation between segment method and profit ###
# Same anova test with the same assumptions
consumer = df.loc[df['Segment'] == "Consumer"]["Profit"].dropna().tolist()
corporate = df.loc[df['Segment'] == "Corporate"]["Profit"].dropna().tolist()
office = df.loc[df['Segment'] == "Home Office"]["Profit"].dropna().tolist()

print(f_oneway(consumer, corporate, office))
# pvalue=0.27975014762339184 - not statistically significant








df.dropna(subset=["Discount", "Profit", "Quantity"], inplace=True)
print("Number of rows: " + str(df.shape[0]))

# problem - might not match up
discount = df["Discount"].to_numpy().reshape(-1, 1)
profit = df["Profit"].to_numpy().reshape(-1, 1)
quantity = df["Quantity"].to_numpy().reshape(-1, 1)



'''
### Correlation between discount and profit ###

# make note about not adding in missing values
regr = linear_model.LinearRegression()
regr.fit(discount, profit)

predictions = regr.predict(discount)

print("Coefficients: \n", regr.coef_)
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(profit, predictions))

plt.scatter(discount, profit, color="black")
plt.title("Profit Plotted Against Discount")
plt.xlabel("Discount")
plt.ylabel("Profit")


plt.plot(discount, predictions, color="blue", linewidth=3)
# how to plot regressin line?

plt.show()
'''






### Correlation between quanity and profit ###

regr = linear_model.LinearRegression()
regr.fit(quantity, profit)

predictions = regr.predict(quantity)

print("Coefficients: \n", regr.coef_)
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(profit, predictions))


plt.scatter(quantity, profit, color="black")
plt.title("Quantity Plotted Against Discount")
plt.xlabel("Quantity")
plt.ylabel("Profit")

plt.plot(quantity, predictions, color="blue", linewidth=3)

plt.show()








### Correlation between discount and quantity ###


# discount correlates with profit
# discount correlates with quantity

# explain why you chose model/test