import os
import re
import emoji
import torch
import typer
import shutil
import warnings
import requests
import subprocess
import numpy as np
import pandas as pd
import tkinter as tk
import ttkbootstrap as ttk
from tqdm import tqdm
from io import BytesIO
from pathlib import Path
from loguru import logger
from langdetect import detect
from tkinter import filedialog
from PIL import Image, ImageStat
from ttkbootstrap.constants import *
from deep_translator import GoogleTranslator
from dateutil.relativedelta import relativedelta
from concurrent.futures import ThreadPoolExecutor
from youtube_trends.config import RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR, KAGGLE_CREDENTIALS_DIR

warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    redownload: bool = typer.Option(False, "--redownload", "-r", help="Download raw dataset. Default value: False."),
    size: str = typer.Option("n", "--size", "-s", help="Specify version of yolov5 to process the dataset (n, s, m, l, x).  Default value: n."),
    weeks: int = typer.Option(0, "--weeks", "-w", help="Number of weeks to use from the raw dataset. Default value: 0 (Complete raw dataset)."),
):
    if size not in {"n", "s", "m", "l", "x"}:
        raise typer.BadParameter("Model must be one of: n, s, m, l, x")

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
    first_process_dataset(size, weeks)

    # -----------------------------------------
    # Creation of processed dataset
    # -----------------------------------------    
    '''
    if not os.path.exists(PROCESSED_DATA_DIR):
        os.makedirs(PROCESSED_DATA_DIR)
        logger.info(f"New folder: {PROCESSED_DATA_DIR}")
    elif output_path.exists():
        output_path.unlink()
    final_process_dataset()
    '''
    
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

# ---------------------------------------------

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

def first_process_dataset(size, weeks):
    """ Initial process of raw dataset. """
    logger.info("Processing raw dataset...")    
    
    df = pd.read_csv(RAW_DATA_DIR / "dataset.csv")

    df['video_published_at'] = pd.to_datetime(df['video_published_at'], errors='coerce').dt.tz_localize(None)
    df['video_trending__date'] = pd.to_datetime(df['video_trending__date'], errors='coerce').dt.tz_localize(None)
    df['days_until_trend'] = (df['video_trending__date'] - df['video_published_at']).dt.days

    df = df.sort_values(by='video_published_at', ascending=False)
    if weeks != 0:
        start_date = df['video_published_at'].iloc[0] - relativedelta(weeks=weeks)
        df = df[df['video_published_at'] >= start_date]
    df = df.dropna()
    df.reset_index(drop=True, inplace=True)

    df = thumbnail_parallel_processing(df, size)
    df = thumbnails_stats_parallel(df) 
    df = process_titles_parallel(df)

    cols_to_drop = ['video_id', 'video_trending__date', 'video_trending_country', 'video_description', 'video_tags', 'video_dimension', 'video_default_thumbnail', 
                    'video_definition', 'video_licensed_content', 'channel_id', 'channel_title', 'channel_description', 'channel_published_at', 'channel_country', 
                    'channel_have_hidden_subscribers', 'channel_video_count', 'channel_localized_title', 'channel_localized_description', 'video_title']
    df = df.drop(cols_to_drop, axis=1)

    df.to_csv(INTERIM_DATA_DIR / 'dataset.csv', index=False)

# ---------------------------------------------

def detect_thumbnail(thumbnail_url, idx, class_names, model, pbar):
    results = model(thumbnail_url)
    class_ids = results.xyxy[0][:, 5].int().tolist()
    detections = np.zeros(len(class_names), dtype=int)
    
    for cls_id in set(class_ids):
        detections[int(cls_id)] = 1
    pbar.update(1)
    
    return idx, detections

# ---------------------------------------------

def  thumbnail_parallel_processing(df, size):
    thumbnail_urls = df['video_default_thumbnail'].values

    match size:
        case "n":
            model = torch.hub.load('ultralytics/yolov5', 'yolov5n', verbose=False).to(device)
        case "s":
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', verbose=False).to(device)
        case "m":
            model = torch.hub.load('ultralytics/yolov5', 'yolov5m', verbose=False).to(device)
        case "l":
            model = torch.hub.load('ultralytics/yolov5', 'yolov5l', verbose=False).to(device)
        case "x":
            model = torch.hub.load('ultralytics/yolov5', 'yolov5x', verbose=False).to(device)
    
    class_names = ['thumbnail_' + name.replace(' ', '_') for name in model.names.values()]
    detections_array = np.zeros((len(thumbnail_urls), len(class_names)), dtype=int)
    
    with tqdm(total=len(thumbnail_urls), desc="Processing thumbnails class") as pbar:
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(detect_thumbnail, thumbnail_url, idx, class_names, model, pbar)
                for idx, thumbnail_url in enumerate(thumbnail_urls)
            ]
            for future in futures:
                idx, detections = future.result()
                detections_array[idx] = detections
    
    detections_df = pd.DataFrame(detections_array, columns=class_names)
    detections_df['video_default_thumbnail'] = df['video_default_thumbnail'].values
    detections_df = detections_df.loc[:, (detections_df != detections_df.iloc[0]).any()]
    df = pd.concat([df, detections_df.iloc[:, :-1]], axis=1)

    return df

# ---------------------------------------------

def thumbnail_stats(thumbnail_url, idx, pbar):
    response = requests.get(thumbnail_url, timeout=10)
    img = Image.open(BytesIO(response.content)).convert('RGB')
    stat = ImageStat.Stat(img)

    brightness = sum(stat.mean) / 3
    contrast = sum(stat.stddev) / 3
    hsv = np.array(img.convert('HSV'))
    saturation = hsv[:, :, 1].mean() / 255

    pbar.update(1)
    return idx, [brightness, contrast, saturation]

# ---------------------------------------------

def thumbnails_stats_parallel(df):
    thumbnail_urls = df['video_default_thumbnail'].values
    stats_array = np.zeros((len(thumbnail_urls), 3), dtype=float)

    with tqdm(total=len(thumbnail_urls), desc="Processing thumbnails stats") as pbar:
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(thumbnail_stats, thumbnail_url, idx, pbar) 
                for idx, thumbnail_url in enumerate(thumbnail_urls)
            ]
            for future in futures:
                idx, stats = future.result()
                stats_array[idx] = stats

    stats_df = pd.DataFrame(stats_array, columns=['brightness', 'contrast', 'saturation'])
    stats_df['video_default_thumbnail'] = thumbnail_urls
    df = pd.concat([df, stats_df.iloc[:, :-1]], axis=1)
    return df

# ---------------------------------------------

def detect_and_translate(title):
    try:
        lang = detect(title)
        translated = GoogleTranslator(source=lang, target='en').translate(title)
    except:
        lang = 'unknown'
        translated = title
    return lang, translated

# ---------------------------------------------

def process_titles_parallel(df, max_workers=8):
    titles = df['video_title'].fillna('').astype(str).tolist()
    languages = []
    translations = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(detect_and_translate, title) for title in titles]
        for future in tqdm(futures, desc="Processing video title"):
            lang, translated = future.result()
            languages.append(lang)
            translations.append(translated)

    df['video_title_language'] = languages
    df['video_title_translated'] = translations
    return df

# ---------------------------------------------------------------------------------------------------------------------------    

def final_process__dataset():
    """ Final process of raw dataset. """
    df = pd.read_csv(INTERIM_DATA_DIR / "dataset.csv")
    print(df.head())


# ---------------------------------------------------------------------------------------------------------------------------    

if __name__ == "__main__":
    app()
