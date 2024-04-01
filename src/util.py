import os
import re

folder_path = '../plot/Vector_field/train_MLPskip_Jac'

# List all files in the folder
files = os.listdir(folder_path)

# Define a regular expression pattern to match the number after "Jacobian"
pattern = r'(\w+_Jacobian)(\d{1,2})'

# Iterate through the files and rename them
for file in files:
    if file.startswith('minRE_Comb_Jacobian'):
        old_name = os.path.join(folder_path, file)
        new_name = re.sub(pattern, r'\1', old_name)
        os.rename(old_name, new_name)

