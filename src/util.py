import os
import re

folder_path = '../plot/Vector_field/train_MLPskip_MSE'

# List all files in the folder
files = os.listdir(folder_path)

# Define a regular expression pattern to match the number after "Jacobian"
pattern = r'\d+'

# Iterate through the files and rename them
for file in files:
    if file.startswith('minRE_Comb'):
        old_name = os.path.join(folder_path, file)
        new_name = re.sub(pattern, "", old_name)
        os.rename(old_name, new_name)

