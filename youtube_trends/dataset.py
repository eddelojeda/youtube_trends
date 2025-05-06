import os
import re
import emoji
import torch
import typer
import shutil
import langid
import isodate
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
from tkinter import filedialog
from PIL import Image, ImageStat
from ttkbootstrap.constants import *
from sklearn.decomposition import PCA
from torchvision import models, transforms
from deep_translator import GoogleTranslator
from langdetect import detect, DetectorFactory
from sklearn.preprocessing import MinMaxScaler
from dateutil.relativedelta import relativedelta
from concurrent.futures import ThreadPoolExecutor
from youtube_trends.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, KAGGLE_CREDENTIALS_DIR

DetectorFactory.seed = 0 
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
    inter_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    redownload: bool = typer.Option(False, "--redownload", "-r", help="Download raw dataset. Default value: False."),
    detect: bool = typer.Option(False, "--detect", "-d", help="Detect objects in thumbnail. Default value: False."),
    extract: bool = typer.Option(False, "--extract", "-e", help="Extract embeddings from thumbnails. Default value: False."),
    size: str = typer.Option("n", "--size", help="Specify version of yolov5 to process the dataset (n, s, m, l, x).  Default value: n."),
    weeks: int = typer.Option(0, "--weeks", help="Number of weeks to use from the raw dataset. Default value: 0 (Complete raw dataset)."),
    threads: int = typer.Option(0, "--threads", help="Number of threads used for parallel data processing. Default value: 0 (Automatic selection based on number of processors)."),
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
    # Creation of processed dataset
    # -----------------------------------------    
    if not os.path.exists(RAW_DATA_DIR / "dataset.csv"):
        print("No dataset available")
    else:
        if not os.path.exists(PROCESSED_DATA_DIR):
            os.makedirs(PROCESSED_DATA_DIR)
            logger.info(f"New folder: {PROCESSED_DATA_DIR}")
        elif inter_path.exists():
            inter_path.unlink()
        process_dataset(detect, extract, size, weeks, threads)
        
# ---------------------------------------------------------------------------------------------------------------------------

def download_dataset():
    """
    Downloads the 'YouTube Trending Videos Global' dataset from Kaggle and saves it to the RAW_DATA_DIR.

    This function uses the Kaggle API to download and unzip the dataset into the predefined RAW_DATA_DIR directory.
    It then renames the extracted CSV file to 'dataset.csv' for standardization. It also shows a progress bar during
    the download process and handles exceptions if the download fails.

    Requirements:
        - Kaggle API credentials must be properly configured (via setup_kaggle_credentials).
        - The 'kaggle' CLI must be installed and accessible.
        - tqdm for progress visualization.

    Raises:
        subprocess.CalledProcessError: If the download process encounters an error.
    """

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
     """
    Set up Kaggle API credentials.

    This function ensures that the required directory for Kaggle credentials exists.
    If it does not, it creates the directory and adds the Kaggle API token by calling `add_kaggle_token()`.
    If the directory exists but the credentials file (`kaggle.json`) is missing, it also calls `add_kaggle_token()`.
    Finally, it sets the environment variable `KAGGLE_CONFIG_DIR` so the Kaggle API knows where to find the credentials.
    """
    
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
    """
    Launches a simple GUI window to prompt the user to select their 'kaggle.json' API token file.

    This function uses a themed Tkinter window to display a file selection prompt.
    The selected file (typically 'kaggle.json') is expected to be the user's Kaggle API token,
    which is required to authenticate and download datasets from Kaggle via the API.

    Once the user selects the file, it should be processed by the `open_file_dialog` function (not shown here).
    """

    root = ttk.Window(themename="litera")
    root.title("Kaggle Token Setup")
    root.geometry("350x120")
    root.resizable(False, False)

    label = ttk.Label(root, text="Select your kaggle.json file", font=("Segoe UI", 11))
    label.pack(pady=(20, 10))
    accept_button = ttk.Button(root, text="Accept", bootstyle=PRIMARY, command=open_file_dialog)
    accept_button.pack()

    root.mainloop()

# ---------------------------------------------

def open_file_dialog():
    """
    Opens a file selection dialog for the user to choose the 'kaggle.json' file.

    - If a file is selected:
        * Creates the Kaggle credentials directory if it doesn't exist.
        * Moves the selected file to the directory as 'kaggle.json'.
        * Logs a success message.
    
    - If no file is selected:
        * Logs an error message.

    Finally, the function closes the Tkinter root window.
    """

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

# ---------------------------------------------------------------------------------------------------------------------------    

def process_dataset(detect, extract, size, weeks, threads, threshold = 0.1):
    """
    Processes the raw YouTube trending dataset by cleaning, transforming, and optionally extracting image features.

    Args:
        detect (bool): If True, performs object detection on thumbnails using a pretrained model.
        extract (bool): If True, extracts embeddings from thumbnails using a neural network.
        size (int): Image size to which thumbnails will be resized for detection/embedding.
        weeks (int): Number of weeks to retain from the most recent publication date. If -1, filters only dates from the day before the most recent video.
        threads (int): Number of threads for parallel processing. If 0, uses all available cores.
        threshold (float, optional): A threshold for filtering video category columns, excluding categories with very low or very high prevalence. Default is 0.1.

    Steps:
        - Loads and filters raw dataset columns.
        - Parses and standardizes date fields.
        - Filters data based on the `weeks` parameter.
        - Extracts temporal features (hour, day of week, days to trend).
        - Computes simple textual features (title length, tag count).
        - Optionally performs thumbnail object detection and feature extraction in parallel.
        - Converts video durations into seconds using parallel threads.
        - Processes video titles (e.g., cleaning or embedding).
        - One-hot encodes video category IDs.
        - Filters out category columns with low or high prevalence using the `threshold` parameter.
        - Saves the processed dataset as a CSV file.

    Output:
        Saves the processed DataFrame to `PROCESSED_DATA_DIR / 'dataset.csv'`.
    """

    logger.info("Processing raw dataset...")    
    
    df = pd.read_csv(RAW_DATA_DIR / "dataset.csv")

    df = df.drop(['video_id', 'video_trending_country', 'video_description', 'video_dimension', 'video_definition', 'video_licensed_content', 
                  'channel_id',  'channel_title', 'channel_published_at', 'channel_description', 'channel_country', 'channel_video_count',
                  'channel_have_hidden_subscribers', 'channel_localized_title', 'channel_localized_description'], axis=1)

    df['video_published_at'] = pd.to_datetime(df['video_published_at'], errors='coerce').dt.tz_localize(None)
    df['video_trending__date'] = pd.to_datetime(df['video_trending__date'], errors='coerce').dt.tz_localize(None)

    df = df.sort_values(by='video_published_at', ascending=False)
    if weeks == -1:
        start_date = df['video_published_at'].iloc[0] - relativedelta(days=1)
        df = df[df['video_published_at'] >= start_dat]
    elif weeks > 0:
        start_date = df['video_published_at'].iloc[0] - relativedelta(weeks=weeks)
        df = df[df['video_published_at'] >= start_date]
    df.reset_index(drop=True, inplace=True)

    df['published_dayofweek'] = df['video_published_at'].dt.dayofweek
    df['published_hour'] = df['video_published_at'].dt.hour
    df['days_to_trend'] = (df['video_trending__date'] - df['video_published_at']).dt.days
    df = df[df['days_to_trend'] >= 0]

    df['video_title_length'] = df['video_title'].str.split().str.len()
    df['video_tag_count'] = df['video_tags'].str.split('|').str.len()
    df['video_tag_count'] = df['video_tag_count'].fillna(0)
    df = df.drop(['video_trending__date', 'video_tags'], axis=1)
    df = df.dropna()

    if threads == 0:
        max_workers = None
    else:
        max_workers = threads
    if detect:
        df = thumbnail_parallel_detect(df, size, max_workers)
    df = thumbnails_parallel_stats(df, max_workers)
    if extract: 
        df = thumbnail_parallel_embeddings(df, max_workers)
    df = df.drop(['video_default_thumbnail'], axis=1)

    durations = df['video_duration'].fillna('').astype(str).tolist()
    with ThreadPoolExecutor(max_workers=max_workers) as executor: 
        duration_secs = list(tqdm(executor.map(convert_duration, durations), total=len(durations), desc="Converting durations"))
    df['video_duration'] = duration_secs

    df = process_titles_parallel(df, max_workers)

    df['video_category_id'] = df['video_category_id'].str.replace(' ', '_')
    df = pd.get_dummies(df, columns=['video_category_id'])
    dummy_cols = [col for col in df.columns if col.startswith('video_category_id_')]
    df[dummy_cols] = df[dummy_cols].astype(int)
    df[dummy_cols] = df[dummy_cols].loc[:, df[dummy_cols].apply(lambda col: col.mean() > threshold and col.mean() < 1.0 - threshold)]
    df = df.dropna()

    df.to_csv(PROCESSED_DATA_DIR / 'dataset.csv', index=False)

# ---------------------------------------------

def convert_duration(duration):
    """
    Converts an ISO 8601 duration string to total seconds.

    Args:
        duration (str): A string representing a duration in ISO 8601 format (e.g., 'PT1H2M10S').

    Returns:
        float: Duration in total seconds. Returns NaN if parsing fails.
    """

    try:
        return isodate.parse_duration(duration).total_seconds()
    except:
        return np.nan

# ---------------------------------------------

def clean_title(title):
    """
    Cleans a video title by removing emojis, punctuation, and extra whitespace.

    Args:
    - title (str): The original title string to be cleaned.

    Returns:
    - str: A cleaned version of the title with no emojis, punctuation, or redundant spaces.
    """

    title = emoji.replace_emoji(title, replace='')
    title = re.sub(r'[^\w\s]', '', title)
    title = re.sub(r'\s+', ' ', title)     

    return title

# ---------------------------------------------

def detect_and_translate(title):
     """
    Detects the language of a given text and translates it to English if it's not already in English.

    Args:
        title (str): The input text (e.g., a video title or description).

    Returns:
        tuple: A tuple (lang, translated_text) where:
            - lang (str): The detected language code (e.g., 'en', 'es', 'fr').
            - translated_text (str): The translated text in English if applicable, or the original text.
    """

    try:
        lang = detect(title)
    except:
        return '', title  
    if lang == 'en':
        return 'en', title
    try:
        translated = GoogleTranslator(source='auto', target='en').translate(title)
        return lang, translated
    except:
        return lang, title

# ---------------------------------------------

def process_titles_parallel(df, max_workers):
     """
    Cleans and processes video titles in parallel using thread-based execution. Fill and cleans missing video titles. 
    Detects the language and translates each cleaned title .

    Args:
        df (pd.DataFrame): DataFrame containing a 'video_title' column.
        max_workers (int): Maximum number of threads to use for language detection and translation.

    Returns:
        pd.DataFrame: The original DataFrame with two additional columns for language and translation.
    """

    titles = df['video_title'].fillna('').astype(str).tolist()
    
    with ThreadPoolExecutor() as executor:
        clean_titles = list(tqdm(executor.map(clean_title, titles), total=len(titles), desc="Cleaning titles"))

    languages = []
    translations = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(detect_and_translate, title) for title in clean_titles]
        for future in tqdm(futures, desc="Processing video title"):
            lang, translated = future.result()
            languages.append(lang)
            translations.append(translated)
    
    df['video_title_language'] = languages
    df['video_title_translated'] = translations

    return df

# ---------------------------------------------

def detect_thumbnail(thumbnail_url, idx, class_names, model, pbar):
    """
    Detects objects in a thumbnail image using a pre-trained model.

    Args:
    - thumbnail_url (str): The URL of the thumbnail image to be processed.
    - idx (int): The index of the current thumbnail, used for tracking.
    - class_names (list): A list of class names for the object detection model.
    - model (torch.nn.Module): The pre-trained object detection model.
    - pbar (tqdm): A progress bar object for tracking the progress of the detection process.

    Returns:
    - idx (int): The index of the current thumbnail.
    - detections (numpy.ndarray): A binary array indicating the presence of detected classes (1 for detected, 0 for not detected).
    """

    results = model(thumbnail_url)
    class_ids = results.xyxy[0][:, 5].int().tolist()
    detections = np.zeros(len(class_names), dtype=int)
    
    for cls_id in set(class_ids):
        detections[int(cls_id)] = 1
    pbar.update(1)
    
    return idx, detections

# ---------------------------------------------

def  thumbnail_parallel_detect(df, size, max_workers, threshold = 0.1):
     """
    Detects objects in the thumbnails of YouTube videos using a YOLOv5 model, with parallel processing.

    Args:
    df (DataFrame): A DataFrame containing video information, including a column 'video_default_thumbnail' with the URLs of the thumbnails.
    size (str): Specifies the size of the YOLOv5 model to be used. Options are:
                - "n": YOLOv5 Nano
                - "s": YOLOv5 Small
                - "m": YOLOv5 Medium
                - "l": YOLOv5 Large
                - "x": YOLOv5 Extra Large
    max_workers (int): The maximum number of parallel workers to use for processing the thumbnails.
    threshold (float): The detection threshold to filter out detections with a low probability. The default value is 0.1.

    Returns:
    DataFrame: A modified version of the input `df` with additional columns for object detections in the thumbnails.
                Each detection column corresponds to a specific object class detected in the thumbnail.
    """

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
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(detect_thumbnail, thumbnail_url, idx, class_names, model, pbar)
                for idx, thumbnail_url in enumerate(thumbnail_urls)
            ]
            for future in futures:
                idx, detections = future.result()
                detections_array[idx] = detections
    
    detections_df = pd.DataFrame(detections_array, columns=class_names)
    detections_df['video_default_thumbnail'] = df['video_default_thumbnail'].values 
    detections_df = detections_df.loc[:, detections_df.apply(lambda col: col.mean() > threshold and col.mean() < 1.0 - threshold)]
    df = pd.concat([df, detections_df.iloc[:, :-1]], axis=1)
    df = df.dropna()

    return df

# ---------------------------------------------

def thumbnail_stats(thumbnail_url, idx, pbar):
    """
    This function computes the brightness, contrast, and saturation of a thumbnail image 
    given its URL. It fetches the image, calculates the required statistics, and returns 
    the results.

    Args:
    thumbnail_url (str): The URL of the thumbnail image.
    idx (int): The index or identifier for the image, used for tracking purposes.
    pbar (tqdm): A progress bar object used to update progress during processing.

    Returns:
    tuple: A tuple containing the index (idx) and a list of three values:
           - Brightness (float): The average brightness of the image.
           - Contrast (float): The average contrast of the image.
           - Saturation (float): The average saturation of the image (normalized between 0 and 1).
           If an error occurs (e.g., image fetch fails), the function returns NaN values for all three stats.
    """

    try:
        response = requests.get(thumbnail_url, timeout=10)
        img = Image.open(BytesIO(response.content)).convert('RGB')
        stat = ImageStat.Stat(img)

        brightness = sum(stat.mean) / 3
        contrast = sum(stat.stddev) / 3
        hsv = np.array(img.convert('HSV'))
        saturation = hsv[:, :, 1].mean() / 255

        pbar.update(1)
        return idx, [brightness, contrast, saturation]
    except Exception:
        pbar.update(1)

        return idx, [np.nan, np.nan, np.nan]

# ---------------------------------------------

def thumbnails_parallel_stats(df, max_workers):
     """
    Computes the brightness, contrast, and saturation statistics for each thumbnail in the provided DataFrame.  The statistics 
    are calculated in parallel using a ThreadPoolExecutor for faster processing. A MinMax scaling is applied to the computed 
    statistics for further normalization

    Args:
    df (pd.DataFrame): The input DataFrame that contains the URLs of the video thumbnails in the column 'video_default_thumbnail'.
    max_workers (int): The maximum number of worker threads to use for parallel processing.

    Returns:
    pd.DataFrame: The original DataFrame with three additional columns for thumbnail statistics:
                  - 'thumbnail_brightness': The brightness statistic of each thumbnail.
                  - 'thumbnail_contrast': The contrast statistic of each thumbnail.
                  - 'thumbnail_saturation': The saturation statistic of each thumbnail.
                  
    Note:
    - The input DataFrame must contain a column named 'video_default_thumbnail' with the URLs of the thumbnails.
    - The `thumbnail_stats` function is expected to calculate the actual brightness, contrast, and saturation statistics for each image.
    """

    urls = df['video_default_thumbnail'].values
    stats_array = np.zeros((len(urls), 3), dtype=float)

    with tqdm(total=len(urls), desc="Computing thumbnail stats") as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(thumbnail_stats, url, idx, pbar)
                for idx, url in enumerate(urls)
            ]
            for future in futures:
                idx, stats = future.result()
                stats_array[idx] = stats

    df_stats = pd.DataFrame(stats_array, columns=["thumbnail_brightness", "thumbnail_contrast", "thumbnail_saturation"])
    
    scaler = MinMaxScaler()
    df_stats_scaled[features] = scaler.fit_transform(df_stats_scaled[features])
    df_stats_scaled.describe()

    df = pd.concat([df, df_stats], axis=1)

    return df

# ---------------------------------------------

def embedding_thumbnail(thumbnail_url, idx, transform, model, pbar):
    """
    This function processes a thumbnail image from a given URL, extracts its features using a pre-trained model,
    and returns the index of the image along with the extracted feature vector.

    Args:
    - thumbnail_url (str): The URL of the thumbnail image to process.
    - idx (int): The index of the current image in the dataset (used for tracking).
    - transform (callable): A transformation function that prepares the image for model input (e.g., resizing, normalization).
    - model (torch.nn.Module): A pre-trained model to extract features from the image (e.g., a CNN model).
    - pbar (tqdm object): A progress bar object to update during processing.

    Returns:
    - idx (int): The index of the image.
    - features (numpy.ndarray): The feature vector extracted from the image.
    
    Note:
    If an error occurs while processing the image, a feature vector of NaN values is returned instead.
    """

    try:
        response = requests.get(thumbnail_url, timeout=10)
        img = Image.open(BytesIO(response.content)).convert('RGB')
        img = transform(img).unsqueeze(0).to(device)  
        with torch.no_grad():
            features = model.features(img)
            features = features.mean([2, 3]).squeeze().cpu().numpy() 
    except Exception as e:
        print(f"Error procesando {thumbnail_url}: {e}")
        features = np.full((1280,), np.nan) 
    pbar.update(1)
    
    return idx, features

# ---------------------------------------------

def thumbnail_parallel_embeddings(df, max_workers, max_components = 40):
    """
    This function extracts embeddings from the thumbnails of videos in the given dataframe. It uses a pre-trained 
    MobileNetV2 model to generate embeddings for each thumbnail URL. The embeddings are then reduced using PCA to 
    retain the most important features, making them suitable for further analysis.

    Args:
    df (pd.DataFrame): DataFrame containing a column 'video_default_thumbnail' with URLs to video thumbnails.
    max_workers (int): Maximum number of workers for parallel processing. 
    max_components (int, optional): The maximum number of PCA components to retain for dimensionality reduction. 
                                    Default is 40.

    Returns:
    pd.DataFrame: The original DataFrame with additional columns containing the reduced thumbnail embeddings.
                  The embeddings are PCA-reduced versions of the original embeddings.
    """

    thumbnail_urls = df['video_default_thumbnail'].values
    
    model = models.mobilenet_v2(pretrained=True)
    model.eval()
    model = model.to(device) 
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    n_samples = len(thumbnail_urls)
    embedding_dim = 1280  
    embeddings_array = np.zeros((n_samples, embedding_dim), dtype=np.float32)

    with tqdm(total=n_samples, desc="Extracting thumbnails embeddings") as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(embedding_thumbnail, url, idx, transform, model, pbar)
                for idx, url in enumerate(thumbnail_urls)
            ]
            for future in futures:
                idx, embedding = future.result()
                embeddings_array[idx] = embedding
    
    valid_rows = ~np.isnan(embeddings_array).any(axis=1) 
    embeddings_array = embeddings_array[valid_rows] 

    pca_complete = PCA().fit(embeddings_array)
    cumulative_variance = np.cumsum(pca_complete.explained_variance_ratio_)
    n_components = np.searchsorted(cumulative_variance, 0.70) + 1 
    n_components = min(n_components, max_components)
    pca = PCA(n_components=n_components)  
    reduced_embeddings = pca.fit_transform(embeddings_array)

    embed_cols = [f'thumb_emb_{i}' for i in range(reduced_embeddings.shape[1])]  
    embeddings_df = pd.DataFrame(reduced_embeddings, columns=embed_cols)
    df = pd.concat([df.reset_index(drop=True), embeddings_df], axis=1)
    df = df.dropna()

    return df

# ---------------------------------------------------------------------------------------------------------------------------    

if __name__ == "__main__":
    app()