import os
import tkinter as tk
from tkinter import filedialog

def parse_filename(filename: str):
    filename = filename.split('/')[3].split('_')
    if filename[0] == 'DQN': model_selection = 1
    elif filename[0] == 'PPO': model_selection = 2
    elif filename[0] == 'A2C': model_selection = 3
    elif filename[0] == 'TQL': model_selection = 4
    else: raise Exception("Invalid Filename")
    return model_selection, float('0.' + filename[1]), bool(filename[2]), int(filename[3]), int(filename[4])

def prompt_zip_file_selection() -> str:
    # Create the root window
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    # Use filedialog to create an open file dialog filtered for .zip files
    file_path = filedialog.askopenfilename(
        title="Select a .zip file",
        initialdir=os.getcwd() + "/training_data",
        filetypes=[("Zip files", "*.zip")],
        defaultextension="*.zip"
    )
    # Destroy the root window
    root.destroy()
    # Return the selected file's name without the .zip extension
    if file_path and file_path.endswith('.zip'):
        date_folder = file_path.split('/')[-2]
        print(os.path.splitext("./training_data/" + date_folder + '/' + os.path.basename(file_path))[0])
        return os.path.splitext("./training_data/" + date_folder + '/' + os.path.basename(file_path))[0]
    return None