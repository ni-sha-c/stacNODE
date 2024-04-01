import os

folder_path = '/path/to/your/folder'

# List all files in the folder
files = os.listdir(folder_path)

# Iterate through the files and rename them
for file_name in files:
    if file_name.startswith('Comb_Jacobian'):
        # Extract the part after "Comb_" and remove the number
        new_file_name = 'Jacobian:' + file_name.split('Comb_Jacobian')[1].split(':')[1]
        # Create the new file path
        new_path = os.path.join(folder_path, new_file_name)
        # Rename the file
        os.rename(os.path.join(folder_path, file_name), new_path)

