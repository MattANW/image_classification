import os
import stat

def rename_files(directory):
    items = os.listdir(directory)
    
    for item in items:
        parts = item.split('.', 1)
        
        if len(parts) == 2 and parts[0].isdigit():
            new_name = parts[1]
            old_path = os.path.join(directory, item)
            new_path = os.path.join(directory, new_name)
            
            os.rename(old_path, new_path)
            print(f"Renamed '{item}' to '{new_name}'")

directory_path = 'images'
rename_files(directory_path)
