import os
import typer
import shutil
import subprocess
import pandas as pd
import tkinter as tk
import ttkbootstrap as ttk
from tqdm import tqdm
from pathlib import Path
from loguru import logger
from tkinter import filedialog
from ttkbootstrap.constants import *
from youtube_trends.config import RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR, KAGGLE_CREDENTIALS_DIR

# ---------------------------------------------------------------------------------------------------------------------------

app = typer.Typer()

@app.command()
def main(
    # -----------------------------------------
    # Default paths and parameters
    # -----------------------------------------
    input_path: Path = RAW_DATA_DIR / "dataset.csv",
    inter_path: Path = INTERIM_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    redownload: bool = typer.Option(False, "--redownload", "-r", help="Download raw dataset"),
):
    # -----------------------------------------
    # Creation of raw dataset
    # -----------------------------------------    
    if redownload:
        if not os.path.exists(RAW_DATA_DIR):
            os.makedirs(RAW_DATA_DIR)
            logger.info(f"New folder: {RAW_DATA_DIR}")
        elif input_path.exists():
            input_path.unlink()
        download_dataset()

    # -----------------------------------------
    # Creation of interim dataset
    # -----------------------------------------    
    if not os.path.exists(INTERIM_DATA_DIR):
        os.makedirs(INTERIM_DATA_DIR)
        logger.info(f"New folder: {INTERIM_DATA_DIR}")
    elif inter_path.exists():
        inter_path.unlink()
    first_process_dataset()

    # -----------------------------------------
    # Creation of processed dataset
    # -----------------------------------------    
    if not os.path.exists(PROCESSED_DATA_DIR):
        os.makedirs(PROCESSED_DATA_DIR)
        logger.info(f"New folder: {PROCESSED_DATA_DIR}")
    elif output_path.exists():
        output_path.unlink()
    final_process_dataset()
    
# ---------------------------------------------------------------------------------------------------------------------------

def download_dataset():
    """ Download dataset from Kaggle. """
    dataset_url = "canerkonuk/youtube-trending-videos-global"
    download_command = [
        "kaggle", "datasets", "download", dataset_url, 
        "--unzip", 
        "--path", str(RAW_DATA_DIR)
    ]
    setup_kaggle_credentials()
    try:
        logger.info("Downloading dataset from Kaggle...")
        with tqdm(total=100, desc="Downloading", ncols=100) as pbar:
            process = subprocess.Popen(download_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            for line in process.stdout:
                if "Downloading" in line and "%" in line:
                    percent = int(line.split("%")[0].split()[-1])
                    pbar.update(percent)
            process.wait()        
            pbar.update(100) 
        shutil.move(RAW_DATA_DIR / "youtube_trending_videos_global.csv", RAW_DATA_DIR / "dataset.csv")
        logger.success("Dataset downloaded successfully.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error downloading the dataset: {e}")
        raise

# ---------------------------------------------------------------------------------------------------------------------------    

def setup_kaggle_credentials():
    """ Setup Kaggle credentials. """
    if not os.path.exists(KAGGLE_CREDENTIALS_DIR):
        os.makedirs(KAGGLE_CREDENTIALS_DIR)
        add_kaggle_token()
    else:
        kaggle_file = KAGGLE_CREDENTIALS_DIR / "kaggle.json"
        if not kaggle_file.exists():
            add_kaggle_token()
    os.environ["KAGGLE_CONFIG_DIR"] = str(KAGGLE_CREDENTIALS_DIR)

def add_kaggle_token():
    """Add Kaggle token to repository."""

    def open_file_dialog():
        file_path = filedialog.askopenfilename(
            title="Select kaggle.json file",
            filetypes=[("JSON files", "*.json")]
        )
        if file_path:
            os.makedirs(KAGGLE_CREDENTIALS_DIR, exist_ok=True)
            shutil.move(file_path, KAGGLE_CREDENTIALS_DIR / "kaggle.json")
            logger.info(f"kaggle.json added to: {KAGGLE_CREDENTIALS_DIR}")
        else:
            logger.error("No file selected.")
        root.quit()

    root = ttk.Window(themename="litera")
    root.title("Kaggle Token Setup")
    root.geometry("350x120")
    root.resizable(False, False)

    label = ttk.Label(root, text="Select your kaggle.json file", font=("Segoe UI", 11))
    label.pack(pady=(20, 10))
    accept_button = ttk.Button(root, text="Accept", bootstyle=PRIMARY, command=open_file_dialog)
    accept_button.pack()

    root.mainloop()

# ---------------------------------------------------------------------------------------------------------------------------    

def first_process__dataset():
    """ Initial process of raw dataset. """
    df = pd.read_csv(RAW_DATA_DIR / "dataset.csv")
    print(df.head())

# ---------------------------------------------------------------------------------------------------------------------------    

def final_process__dataset():
    """ Final process of raw dataset. """
    df = pd.read_csv(INTERIM_DATA_DIR / "dataset.csv")
    print(df.head())


# ---------------------------------------------------------------------------------------------------------------------------    

if __name__ == "__main__":
    app()
