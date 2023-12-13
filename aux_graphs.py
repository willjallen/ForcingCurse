# %%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as ticker
import matplotlib.animation as animation
import seaborn as sns
import os
import shutil
import math

from scipy.stats import pareto

from data import CitiesData, TreesData, BooksData, SmallBodiesData

# %%
# Output Directory Setup
#================================================================
#region
OUTPUT_DIRECTORY = 'out/auxiliary'

if os.path.exists(OUTPUT_DIRECTORY):
	shutil.rmtree(OUTPUT_DIRECTORY)
 
os.makedirs(OUTPUT_DIRECTORY)

plt_cnt = 0

def save_fig(plt, name):
	global plt_cnt
	plt.savefig(f'{OUTPUT_DIRECTORY}/{plt_cnt}_{name}')
	plt_cnt += 1
#endregion
# %%
# Plot notator wrapper function
#================================================================
#region
def notate_plot(plt: plt, data_source="", website="wallen.me/projects/modeling-wealth", note="", margin=0.18):
	# Adjust the bottom margin to make space for the note
	plt.subplots_adjust(bottom=margin)

	extra_note = f"Note: {note}" if note else ""

	# Add the data source and website URL to the plot
	note_text = ""
	if website:
		note_text = f"Data Source: {data_source} \n More info: {website}\n{extra_note}"
	else:
		note_text = f"Data Source: {data_source}"
	plt.text(0.95, 0.04, note_text, 
			 ha='right', va='center', transform=plt.gcf().transFigure, fontsize=9, alpha=0.7)
#endregion
# %%
# Graph Styling
#================================================================
#region
# Set the Seaborn style
sns.set_style("darkgrid")

# Set global defaults for matplotlib
plt.rcParams['savefig.dpi'] = 300  # set the DPI for saved figures

# Use Seaborn's blue color palette
sns.set_palette(sns.dark_palette("#82408c", reverse=False))  
#endregion
# %%
# Importing Data
#================================================================
#region
cities_data = CitiesData()
cities_data.load()
cities_df = cities_data.get_cities_data()

small_bodies_data = SmallBodiesData()
small_bodies_data.load()
small_bodies_df = small_bodies_data.get_small_bodies_data()

trees_data = TreesData()
trees_data.load()
trees_df = trees_data.get_trees_data()

books_data = BooksData()
books_data.load()
books_df = books_data.get_books_data()
#endregion

# %%
# Diameters of observed small bodies in solar system
#================================================================
# https://ssd.jpl.nasa.gov/

plt.figure(figsize=(12, 8))
plt.hist(small_bodies_df['diameter'], bins=100)

# Title and labels
plt.title('Diameter of Small Bodies in the Solar System', fontsize=16)
plt.ylabel('Frequency')
plt.xlabel('Diameter (km)', labelpad=30)


# y-axis
# locs, labels = plt.yticks()  # Get current y-axis tick locations and labels
# plt.yticks(locs, [f"{x*1e-12:.1f}T" for x in locs])  # Set new labels in trillions

plt.yscale('log')  # Log scale for y axis
# Plot properties
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()

# Notate
notate_plot(plt, data_source="ssd.jpl.nasa.gov", website="")

# Save
save_fig(plt, 'mass_solar_system.png')


# %%
# Population in cities
#================================================================
# https://simplemaps.com/data/world-cities
cities_df.sort_values(by=['population'])

plt.figure(figsize=(12, 8))
plt.hist(cities_df['population'], bins=100)

# Title and labels
plt.title('Population of Cities', fontsize=16)
plt.ylabel('Frequency')
plt.xlabel('Population', labelpad=30)



plt.yscale('log')  # Log scale for y axis

# x-axis
locs, labels = plt.xticks()  # Get current y-axis tick locations and labels
plt.xticks(locs, [f"{x*1e-6:.1f}M" for x in locs])  # Set new labels in trillions
plt.xlim(left=0)

# Plot properties
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()

# Notate
notate_plot(plt, data_source="simplemaps.com/data/world-cities", website="")

# Save
save_fig(plt, 'cities_dist.png')
# %%
# Height of Trees in Alabama
#================================================================

# https://apps.fs.usda.gov/
# filtered_trees_df = trees_df[trees_df['HT'] < 200]
plt.figure(figsize=(12, 8))
plt.hist(trees_df['DIA'], bins=100)

# Title and labels
plt.title('Diamater of Trees in Alabama\'s Forests', fontsize=16)
plt.ylabel('Frequency')
plt.xlabel('Diamater (in)', labelpad=30)

# plt.yscale('log')  # Log scale for y axis

# x-axis
# locs, labels = plt.xticks()  # Get current y-axis tick locations and labels
# plt.xticks(locs, [f"{x*1e-6:.1f}M" for x in locs])  # Set new labels in trillions
plt.xlim(left=0, right=50)

# Plot properties
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()

# Notate
notate_plot(plt, data_source="apps.fs.usda.gov", website="")

# Save
save_fig(plt, 'trees.png')

# %%
# Book pages
#================================================================

plt.figure(figsize=(12, 8))
plt.hist(books_df['pages'], bins=100)

# Title and labels
plt.title('Number of Pages in Books', fontsize=16)
plt.ylabel('Frequency')
plt.xlabel('Pages', labelpad=30)



plt.yscale('log')  # Log scale for y axis

# x-axis
# locs, labels = plt.xticks()  # Get current y-axis tick locations and labels
# plt.xticks(locs, [f"{x*1e-6:.1f}M" for x in locs])  # Set new labels in trillions
plt.xlim(left=0)

# Plot properties
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()

# Notate
notate_plot(plt, data_source="kaggle.com/datasets/mdhamani/goodreads-books-100k")

# Save
save_fig(plt, 'book_pages.png')

# %%
from PIL import Image
import matplotlib.pyplot as plt

# Load the images
img1 = Image.open("./out/auxiliary/0_mass_solar_system.png")
img2 = Image.open("./out/auxiliary/1_cities_dist.png")
img3 = Image.open("./out/auxiliary/2_trees.png")
img4 = Image.open("./out/auxiliary/3_book_pages.png")

# Determine the size for the composite image (max dimensions of individual images)
max_width = max(img1.width, img2.width, img3.width, img4.width)
max_height = max(img1.height, img2.height, img3.height, img4.height)

# Create a new blank image with a white background
composite = Image.new('RGB', (max_width * 2, max_height * 2), color='white')

# Paste the images into the composite image
composite.paste(img1.resize((max_width, max_height)), (0, 0))
composite.paste(img2.resize((max_width, max_height)), (max_width, 0))
composite.paste(img3.resize((max_width, max_height)), (0, max_height))
composite.paste(img4.resize((max_width, max_height)), (max_width, max_height))

# Save the composite image
composite_path = "./out/auxiliary/composite_image.png"
composite.save(composite_path)
# %%

# # %%
# # Mass in solar system
# #================================================================
# # https://articles.adsabs.harvard.edu/pdf/1977Ap%26SS..51..153W 
# # https://nssdc.gsfc.nasa.gov/planetary/factsheet/
# # Adjusting the graph according to the new specifications

# updated_planetary_data = {
#     'Mercury': 0.053,
#     'Venus': 0.815,
#     'Earth': 1.0,
#     'Mars': 0.107,
#     'Asteroids': 0.0005, 
#     'Jupiter': 318,
#     'Saturn': 95,
#     'Uranus': 14.6,
#     'Neptune': 17.2
# }

# df_planets = pd.DataFrame(sorted(updated_planetary_data.items(), key=lambda item: item[1], reverse=True), columns=['Body', 'Mass'])

# plt.figure(figsize=(12, 8))
# plt.bar(df_planets['Body'], df_planets['Mass'])

# # Title and labels
# plt.title('Mass of Solar System Bodies', fontsize=16)
# plt.ylabel('Mass (Earth Masses)')
# plt.xlabel('Body', labelpad=30)


# # y-axis
# # locs, labels = plt.yticks()  # Get current y-axis tick locations and labels
# # plt.yticks(locs, [f"{x*1e-12:.1f}T" for x in locs])  # Set new labels in trillions

# # plt.yscale('log')  # Log scale for y axis
# # Plot properties
# plt.grid(True, which='both', linestyle='--', linewidth=0.5)
# plt.tight_layout()

# # Notate
# notate_plot(plt, data_source="nssdc.gsfc.nasa.gov/planetary/factsheet")

# # Save
# save_fig(plt, 'mass_solar_system.png')