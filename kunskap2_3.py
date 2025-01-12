# 
# # Pandas
# Read "10 minutes to Pandas": https://pandas.pydata.org/docs/user_guide/10min.html before solving the exercises.
# We will use the data set "cars_data" in the exercises below. 

# 
# Importing Pandas. 
import pandas as pd

# 
# ### Explain what a CSV file is.

# 
### CSV file is a comma seperated values file, which allows data to be saved in a table structured format in plain text. Each line of the CSV file is a data record and each record consist one or more fields seperated by commas. CSVs look like a spreadsheet but with a csv-extension for example template csv. 
### CSV file is a simple text file used to store tabular data and each row is seperated by a newline and values are seperated by a comma. 
### A CSV file is easy to read and write and is widely used in data storage. 

# 
# ### Load the data set "cars_data" through Pandas. 

# 
# When reading in the data, either you have the data file in the same folder as your python script
# or in a seperate folder.

# Code below can be ran if you have the data file in the same folder as the script
cars = pd.read_csv("cars_data.csv")

# Code below can be ran if you have the data file in another script. 
# Notice, you must change the path according to where you have the data in your computer. 
# pd.read_csv(r'C:\Users\Antonio Prgomet\Documents\03_nbi_yh\korta_yh_kurser\python_för_ai\kunskapskontroll_1\cars_data.csv')

# 
# ### Print the first 10 rows of the data. 

# 
print(cars.head(10))

# 
# ### Print the last 5 rows. 

# 
print(cars.tail())

# 
# ### By using the info method, check how many non-null rows each column have. 

# 
cars.info()

# 
# ### If any column has a missing value, drop the entire row. Notice, the operation should be inplace meaning you change the dataframe itself.

# 
cars.dropna(inplace=True)

# 
cars.info()

# 
# ### Calculate the mean of each numeric column. 

# 
mean_values = cars.select_dtypes(include='number').mean()
print(mean_values)

# 
# ### Select the rows where the column "company" is equal to 'honda'. 

# 
honda_company_cars = cars[cars['company'] == 'honda']
print(honda_company_cars)

#
# ### Sort the data set by price in descending order. This should *not* be an inplace operation. 

# 
sort_cars_by_price_desc = cars.sort_values(by='price', ascending=False)
print(sort_cars_by_price_desc)

# 
# ### Select the rows where the column "company" is equal to any of the values (audi, bmw, porsche).

# 
select_cars_by_company = cars[cars['company'].isin(['audi', 'bmw', 'porsche'])]
print(select_cars_by_company)

# 
# ### Find the number of cars (rows) for each company. 

# 
number_of_cars_by_company = cars.groupby('company').size()
print(number_of_cars_by_company)

# 
# ### Find the maximum price for each company. 

# 
maximum_price_by_company = cars.groupby('company')['price'].max()
print(maximum_price_by_company)