import pandas as pd
import numpy as np
import re

import copy



# https://www.federalreserve.gov/releases/z1/dataviz/dfa/distribute/chart/#range:1989.3,2023.2;quarter:135;series:Net%20worth;demographic:networth;population:all;units:levels
class FedData():
   # Define population sizes for each category
	POPULATION_SIZES = {
		'Bottom50': 0.5,
		'Next40': 0.4,
		'Next9': 0.09,
		'RemainingTop1': 0.009,
		'TopPt1': 0.001
	}

	# Define the population percentiles for each category
	PERCENTILES = {
		'Bottom50': (0, 50),
		'Next40': (50, 90),
		'Next9': (90, 99),
		'RemainingTop1': (99, 99.99),
		'TopPt1': (99.99, 100)
	}

	PERCENTILES_STR = {
		'Bottom50': '0-50',
		'Next40': '50-90',
		'Next9': '90-99',
		'RemainingTop1': '99-99.99',
		'TopPt1': '99.99-100'
	}

	PERCENTILES_STR_LIST = ['0-50', '50-90', '90-99', '99-99.99', '99.99-100']

 
	def __init__(self):
		self.loaded = False
	
	def load(self):
		print("Loading FED net worth data...")
		self.df = pd.read_csv("data/FED/dfa-networth-levels.csv")

		# Adjust the date format and convert to datetime
		self.df['Date'] = self.df['Date'].str.replace(':', '-').astype('period[Q]')

		# Pivot the dataframe
		self.df_net_worth = self.df.pivot(index='Date', columns='Category', values='Net worth')
  
		# Renormalize units to single dollars
		self.df_net_worth *= 1_000_000 

		self.loaded = True
		print("FED net worth data loaded")
	
	def get_net_worth_data(self):
		if not self.loaded:
			raise Exception("Data not loaded. Call the 'load' method first.") 
		return self.df_net_worth.copy()


class PSIDData():
	def __init__(self):
		self.loaded = False
  
	def load(self, cpi_adjust: bool, equivalence_scale_adjust: bool, target_year=2022):
		print("Loading PSID household wealth data...")
  
  
		if cpi_adjust:
			oecd_data = OECDData()
			oecd_data.load()
			cpi_data = oecd_data.get_cpi_data()
   
			# Find the CPI value for the target year
			cpi_target_year = cpi_data[cpi_data['TIME'] == target_year]['Value'].iloc[0]
   
			# Calculate multipliers and convert year to string in the dictionary
			self.cpi_multiplier_dict = {
				str(year): cpi_target_year / cpi_value
				for year, cpi_value in zip(cpi_data['TIME'], cpi_data['Value'])
			}

		# Load the data labels
		with open("data/PSID/data_labels.txt", 'r') as file:
			contents = file.readlines()	
   
		# Convert the data labels to a dictionary
		self.variables_dict = self.parse_to_dict(contents)
  
		# Extract the years from the data labels
		self.year_dict = {var: self.extract_year(var, label) for var, label in self.variables_dict.items() if self.extract_year(var, label)}
 
		# Load the main data
		self.household_wealth_data_df = pd.read_csv("data/PSID/household-wealth-data.csv")

		# Initialize an empty dictionary to hold the dataframes for each year
		self.household_wealth_year_dfs = {}

		# Iterate over each year and create a dataframe
		for year in set(self.year_dict.values()):
			# Find all columns for this year
			columns_for_year = [var for var, yr in self.year_dict.items() if yr == year]

			# Identify the index column and the wealth columns
			index_column = next((col for col in columns_for_year if 'INTERVIEW' in self.variables_dict[col]), None)
			imp_wealth_column = next((col for col in columns_for_year if 'IMP WEALTH' in self.variables_dict[col]), None)
			acc_wealth_column = next((col for col in columns_for_year if 'ACC WEALTH' in self.variables_dict[col]), None)
			num_family_unit_column = next((col for col in columns_for_year if '# IN FU' in self.variables_dict[col]), None) 

			# Create a dataframe with the relevant columns
			if index_column and imp_wealth_column and acc_wealth_column and num_family_unit_column:
				year_df = self.household_wealth_data_df[[index_column, imp_wealth_column, acc_wealth_column, num_family_unit_column]].copy()
				year_df.set_index(index_column, inplace=True)
				year_df.index.name = 'FAMILY ID'
				year_df.columns = ['IMP WEALTH W/ EQUITY', 'ACC WEALTH W/ EQUITY', '# IN FU']
	
				# Remove empty rows (NaN)
				year_df = year_df.dropna(axis=0, how='any')	
	
				# Adjust for inflation if cpi_adjust is True
				if cpi_adjust:
					multiplier = self.cpi_multiplier_dict[year]
					year_df['IMP WEALTH W/ EQUITY'] *= multiplier
				'''
				A. B. Atkinson, L. Rainwater, T. M. Smeeding, Income Distribution in OECD Countries: Evidence
				from Luxembourg Income Study, Organization for Economic Co-operation and Development, Paris,
				1995. 
				''' 
				# Net household wealth is divided by the square root of the number of household members
				if equivalence_scale_adjust:
					year_df['IMP WEALTH W/ EQUITY'] /= np.sqrt(year_df['# IN FU'])
	 
				self.household_wealth_year_dfs[year] = year_df

			else:
				print("Error: missing column", index_column, imp_wealth_column, acc_wealth_column, num_family_unit_column)

		self.loaded = True
		print("PSID household wealth data loaded")
  
	def get_household_wealth_data(self):
		if not self.loaded:
			raise Exception("Data not loaded. Call the 'load' method first.") 
		return copy.deepcopy(self.household_wealth_year_dfs)

	# Function to parse the labels and convert it to a dictionary
	def parse_to_dict(self, lines):
		variable_dict = {}
		for line in lines:
			# Skip lines that don't contain variable information
			if not line.strip() or "Variable" in line or "****" in line:
				continue
			# Split the line into variable and label
			parts = line.split(maxsplit=1)
			if len(parts) == 2:
				var, label = parts
				variable_dict[var.strip()] = label.strip()
		return variable_dict

	# Function to extract year from the label
	def extract_year(self, var, label):
		# Try to find a 4-digit year in the label 
		label_year_match = re.search(r'\b(19|20)\d{2}\b', label)
		if label_year_match:
			return label_year_match.group()
		# Return None if no year found
		return None


class OECDData():
	def __init__(self):
		self.loaded = False
		self.cpi_df = None

	def load(self):
		print("Loading OECD CPI data...")
		df = pd.read_csv('data/OECD/USA-CPI-1980-2022.csv')
		self.cpi_df = df[['TIME', 'Value']].copy()
		self.loaded = True
		print("OECD CPI data loaded")
  

	def get_cpi_data(self):
		if not self.loaded:
			raise Exception("Data not loaded. Call the 'load' method first.")
		return self.cpi_df.copy()

class CitiesData():
	def __init__(self):
		self.loaded = False
	
	def load(self):
		print("Loading cities data...")
		self.df_cities = pd.read_csv("data/CITIES/worldcities.csv")

		self.loaded = True
		print("Cities data loaded")
	
	def get_cities_data(self):
		if not self.loaded:
			raise Exception("Data not loaded. Call the 'load' method first.") 
		return self.df_cities.copy()

class SmallBodiesData():
	def __init__(self):
		self.loaded = False
	
	def load(self):
		print("Loading small bodies data...")
		self.df_small_bodies = pd.read_csv("data/SMALL_BODIES/sbdb.csv")

		self.loaded = True
		print("Small bodies data loaded")
	
	def get_small_bodies_data(self):
		if not self.loaded:
			raise Exception("Data not loaded. Call the 'load' method first.") 
		return self.df_small_bodies.copy()



class TreesData():
	def __init__(self):
		self.loaded = False
		self.cleaned_file_path = "data/TREES/CLEANED_AL_TREE.csv"
		self.original_file_path = "data/TREES/AL_TREE.csv"
	def load(self):

		try:
			# First, try to load the cleaned data
			print("Trying to load cleaned trees data...")
			self.df_trees = pd.read_csv(self.cleaned_file_path)
			print("Cleaned trees data loaded")
		except FileNotFoundError:
			# If the cleaned data is not found, load and clean the original data
			print("Cleaned data not found, loading and cleaning original data...")
			self.df_trees = pd.read_csv(self.original_file_path)
			self.loaded = True
			self.clean()
			print("Original data cleaned. Try loading again.")
			return
		self.loaded = True



	def clean(self, columns=["HT", "DIA"]):
		if not self.loaded:
			raise Exception("Data not loaded. Call the 'load' method first.")
		cleaned_data = self.df_trees[columns]
		cleaned_data = cleaned_data.dropna(how='all')	
		cleaned_data.to_csv(self.cleaned_file_path, index=False)
		print("Data cleaned and saved as CLEANED_AL_TREE.csv")

	
	def get_trees_data(self):
		if not self.loaded:
			raise Exception("Data not loaded. Call the 'load' method first.") 
		return self.df_trees.copy()

class BooksData():
	def __init__(self):
		self.loaded = False
		self.cleaned_file_path = "data/BOOKS/CLEANED_GoodReads_100k_books.csv"
		self.original_file_path = "data/BOOKS/GoodReads_100k_books.csv"

	def load(self):

		try:
			# First, try to load the cleaned data
			print("Trying to load cleaned books data...")
			self.df_books = pd.read_csv(self.cleaned_file_path)
			print("Cleaned books data loaded")
		except FileNotFoundError:
			# If the cleaned data is not found, load and clean the original data
			print("Cleaned data not found, loading and cleaning original data...")
			try:
				self.df_books = pd.read_csv(self.original_file_path)
			except FileNotFoundError:
				raise Exception("Original data file not found. Please check the file path.")
			except pd.errors.ParserError:
				raise Exception("Error parsing the CSV file. Please check the file format.")

			self.loaded = True
			self.clean()
			print("Original data cleaned. Try loading again.")
			return

		self.loaded = True

	def clean(self, columns=["pages"]):
		if not self.loaded:
			raise Exception("Data not loaded. Call the 'load' method first.")
		# Select the specified columns
		cleaned_data = self.df_books[columns]
		cleaned_data = cleaned_data.dropna(how='all')	
		# Save to a new CSV file
		cleaned_data.to_csv(self.cleaned_file_path, index=False)
		print("Data cleaned and saved as CLEANED_GoodReads_100k_books.csv")

  
	def get_books_data(self):
		if not self.loaded:
			raise Exception("Data not loaded. Call the 'load' method first.") 
		return self.df_books.copy()


# if __name__ == "__main__":
#     cities_data = CitiesData()
#     cities_data.load()


# class ForbesData():
# 	def __init__(self):
# 		self.loaded = False
		
# 	def load(self):
# 		pass

# 	def get_inflation_data():
# 		return self.inflation_df.copy()

# fed_data = FedData()
# fed_data.load()
# print(fed_data.get_net_worth_data())


