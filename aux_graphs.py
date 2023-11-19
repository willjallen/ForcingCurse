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

#================================================================
# Output Directory Setup
#================================================================

OUTPUT_DIRECTORY = 'out/auxiliary'

if os.path.exists(OUTPUT_DIRECTORY):
	shutil.rmtree(OUTPUT_DIRECTORY)
 
os.makedirs(OUTPUT_DIRECTORY)

plt_cnt = 0

def save_fig(plt, name):
	global plt_cnt
	plt.savefig(f'{OUTPUT_DIRECTORY}/{plt_cnt}_{name}')
	plt_cnt += 1
 

#================================================================
# Plot notator wrapper function
#================================================================

def notate_plot(plt: plt, data_source="", website="wallen.me/projects/modeling-wealth", note="", margin=0.18):
	# Adjust the bottom margin to make space for the note
	plt.subplots_adjust(bottom=margin)

	extra_note = f"Note: {note}" if note else ""

	# Add the data source and website URL to the plot
	note_text = f"More info: {website}\n{extra_note}"
	plt.text(0.95, 0.04, note_text, 
			 ha='right', va='center', transform=plt.gcf().transFigure, fontsize=9, alpha=0.7)


#================================================================
# Graph Styling
#================================================================

# Set the Seaborn style
sns.set_style("darkgrid")

# Set global defaults for matplotlib
plt.rcParams['savefig.dpi'] = 300  # set the DPI for saved figures

# Use Seaborn's blue color palette
sns.set_palette(sns.dark_palette("#69d", reverse=False))  
 
