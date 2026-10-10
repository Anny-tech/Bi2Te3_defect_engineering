#!/usr/bin/python

import numpy as np
import pandas as pd
import re
import os
import pymatgen.core.structure as st
import fnmatch
import rdfpy as rdf
from pymatgen.transformations.standard_transformations import SupercellTransformation as cst
from joblib import Parallel, delayed
personal_path= '/rel_pdf_analysis/'

structures = []
f_names = []

# Regular expression pattern to match the filenames and extract the numbers
pattern = re.compile(r'struc_(\d+)_rel\.cif')

# Iterate over the files in the directory
for filename in os.listdir(personal_path):
    match = pattern.match(filename)
    if match:
        number = int(match.group(1))
        structures.append(number)
        f_names.append(filename)

def data_builder(path, file_type, structure_id):
    file_path = os.path.join(path, file_type)
    # Load structure from file
    structure = st.Structure.from_file(file_path)

    # Make supercell
    structure.make_supercell(10)

    # Apply transformation and add noise
    tfms = cst().apply_transformation(structure)
    coords = tfms.cart_coords
    noise = np.random.normal(loc=0.0, scale=0.05, size=coords.shape)
    coords += noise

    # Calculate RDF
    g_r, radii = rdf.rdf(coords, dr=0.05)
    print('G(r):', g_r)
    print('\n')
    print('radii ($\AA$): ', radii)
    print('\n')
    print('\n')

    return g_r, radii

# Use Parallel and delayed to call data_builder in parallel
results = Parallel(n_jobs=4)(delayed(data_builder)(personal_path, f_name, structure_id) for f_name, structure_id in zip(f_names, structures))

# Separate the results into data_unrel and rad
data_rel, rad = zip(*results)

# Convert lists to dataframes
df_rel_v1 = pd.DataFrame(data_rel)

rad_df = pd.DataFrame(rad)

# Calculate the average of each column in rad_df
rad_averages = rad_df.mean().tolist()

# Filter rad_averages to only include values between 0 and 10
filtered_rad_averages = [avg for avg in rad_averages if 0 <= avg <= 10]

# Generate column names using filtered_rad_averages
filtered_column_names = [f'v_{avg:.2f}' for avg in filtered_rad_averages]


# Restrict df_unrel_v1 to only include columns up to the length of filtered_column_names
if len(df_rel_v1.columns) > len(filtered_column_names):
    df_rel_v1 = df_rel_v1.iloc[:, :len(filtered_column_names)]

# Assign filtered column names to df_unrel_v1
df_rel_v1.columns = filtered_column_names


# Generate column names
#column_names = [f'v_{i}' for i in rad[0]]  # Using rad[0] to get the column names

# Assign column names to the DataFrame
#df_unrel_v1.columns = column_names
df_rel_v1['structures'] = structures
df_rel_v1.to_csv('/rel_pdf_analysis/data_fin.csv', index=False)
