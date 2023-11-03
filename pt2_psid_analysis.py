import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shutil
from io import StringIO
from data import FedData, PSIDData
from utils.helper import calculate_percentiles

#================================================================
# Output Directory Setup
#================================================================

OUTPUT_DIRECTORY = 'out/pt2'

if os.path.exists(OUTPUT_DIRECTORY):
	shutil.rmtree(OUTPUT_DIRECTORY)
 
os.makedirs(OUTPUT_DIRECTORY)

plt_cnt = 0

def save_fig(plt, name):
	global plt_cnt
	plt.savefig(f'{OUTPUT_DIRECTORY}/{plt_cnt}_{name}')
	plt_cnt += 1
 
#================================================================
# Graph Styling
#================================================================

# Set the Seaborn style
sns.set_style("darkgrid")

# Set global defaults for matplotlib
plt.rcParams['savefig.dpi'] = 300  # set the DPI for saved figures

# Use Seaborn's green color palette
sns.set_palette(sns.dark_palette("#69d"))  
 
 
#================================================================
# Importing Data
#================================================================

psid_data = PSIDData()
psid_data.load(cpi_adjust=True, target_year=2022)
psid_household_wealth_dict = psid_data.get_household_wealth_data()


#================================================================
# Graph of proportion of total wealth vs wealth for a specific year (2005)
#================================================================

chosen_period = '2005'
psid_household_wealth_chosen_period_df = psid_household_wealth_dict['2005']

# Calculate the percentiles
wealth_percentiles = calculate_percentiles(psid_household_wealth_chosen_period_df, 'IMP WEALTH W/ EQUITY', 0.01)

percentiles_df = pd.DataFrame(list(wealth_percentiles.items()), columns=['Percentile', 'Wealth'])

plt.figure(figsize=(10, 6))
plt.scatter(percentiles_df['Percentile'], percentiles_df['Wealth'])

# Set the title and labels
plt.title("Wealth Distribution 2005")
plt.xlabel('Percentile')
plt.ylabel('Wealth') 
plt.yscale('symlog') 
save_fig(plt, "wealth_distribution_2005_scatter")

