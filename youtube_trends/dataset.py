import os
import re
import time
import emoji
import torch
import typer
import string
import joblib
import random
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
from urllib3.util.retry import Retry
from sklearn.decomposition import PCA
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from requests.adapters import HTTPAdapter
from torchvision import models, transforms
from deep_translator import GoogleTranslator
from langdetect import detect, DetectorFactory
from dateutil.relativedelta import relativedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from concurrent.futures import ThreadPoolExecutor, as_completed
from youtube_trends.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, KAGGLE_CREDENTIALS_DIR

DetectorFactory.seed = 42 
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
    output_train_path: Path = PROCESSED_DATA_DIR / "train_dataset.csv",
    output_val_path: Path = PROCESSED_DATA_DIR / "val_dataset.csv",
    output_test_path: Path = PROCESSED_DATA_DIR / "test_dataset.csv",
    redownload: bool = typer.Option(False, "--redownload", "-r", help="Download raw dataset. Default value: False."),
    vectorize: bool = typer.Option(False, "--vectorize", "-v", help="Vectorize and detect language of the video title. Default value: False."),
    translate: bool = typer.Option(False, "--translate", "-t", help="translate the video titles to english. Default value: False."),
    detect: bool = typer.Option(False, "--detect", "-d", help="Detect objects in thumbnail. Default value: False."),
    stats: bool = typer.Option(False, "--stats", "-s", help="Compute the stats brightness, contrast and saturation in thumbnail. Default value: False."),
    embed: bool = typer.Option(False, "--embed", "-e", help="Extract embeddings from thumbnails. Default value: False."),
    size: str = typer.Option("n", "--size", help="Specify version of yolov5 to process the dataset (n, s, m, l, x).  Default value: n."),
    weeks: int = typer.Option(0, "--weeks", help="Number of weeks to use from the raw dataset. Default value: 0 (Complete raw dataset)."),
    days: int = typer.Option(0, "--days", help="Number of days to use from the raw dataset. Default value: 0 (Complete raw dataset)."),
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
        process_dataset(vectorize, translate, detect, stats, embed, size, weeks, days, threads)
        
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

def process_dataset(vectorize, translate, detect, stats, embed, size, weeks, days, threads, threshold = 0.1):
    """
    Processes the raw YouTube trending dataset by performing extensive cleaning, feature engineering, and optional image and text 
    feature extraction, followed by dataset splitting and saving.

    The function executes the following steps:
    - Loads and cleans raw columns from the dataset.
    - Parses and standardizes date and time fields.
    - Filters data to retain videos from the most recent `weeks`.
    - Extracts temporal features: publishing weekday, hour, and time to trend.
    - Computes basic textual features such as title length and number of tags.
    - Converts video durations into seconds using multithreading.
    - Optionally runs object detection on thumbnails using YOLOv5.
    - Optionally computes brightness, contrast, and saturation statistics for thumbnails.
    - Optionally extracts embeddings from thumbnails using a pretrained MobileNetV2.
    - Reduces thumbnail embeddings dimensionality via PCA fitted on training data.
    - Optionally cleans, detects language, and translates video titles to English.
    - Splits the dataset into training (70%), validation (15%), and test (15%) sets.
    - Scales thumbnail image statistics using MinMaxScaler fitted on training data.
    - Optionally vectorizes titles using TF-IDF and encodes language using one-hot encoding.
    - Applies one-hot encoding to `video_category_id` and filters by prevalence threshold.
    - Optionally reduces video category encoding dimensionality using PCA.
    - Drops unnecessary columns and missing data.
    - Saves the processed train/val/test datasets as CSV files.
    - Saves all fitted transformers (scalers, encoders, vectorizer, PCA models) as `.pkl` files.

    Args:
        vectorize (bool): Whether to apply TF-IDF vectorization and language encoding on video titles.
        translate (bool): Whether to clean, detect language, and translate titles to English.
        detect (bool): Whether to apply object detection to video thumbnails.
        stats (bool): Whether to compute and scale thumbnail image statistics.
        embed (bool): Whether to extract MobileNetV2 embeddings from thumbnails.
        size (str): YOLO model size used for object detection ('n', 's', 'm', 'l', 'x').
        weeks (int): Number of recent weeks of data to keep. If -1, keeps only data from the day before the latest video.
        days (int): Number of recent days of data to keep. If -1, keeps only data from the day before the latest video.
        threads (int): Number of threads for parallel processing. If 0, uses all available cores.
        threshold (float, optional): Prevalence threshold to filter one-hot encoded `video_category_id` columns. Default is 0.1.

    Returns:
        None. Saves the following files:
        - Train/validation/test datasets in `PROCESSED_DATA_DIR` as CSVs.
        - Fitted transformers in `MODELS_DIR` as `.pkl` files:
            - 'stats_scaler.pkl'
            - 'title_vectorizer.pkl'
            - 'title_encoder.pkl'
            - 'category_encoder.pkl'
            - 'category_pca.pkl'
            - 'thumbnail_pca.pkl'
    """

    logger.info("Processing raw dataset...")    
    
    df = pd.read_csv(RAW_DATA_DIR / "dataset.csv")

    df = df.drop(['video_id', 'video_trending_country', 'video_description', 'video_dimension', 'video_definition', 'video_licensed_content', 
                  'channel_id',  'channel_title', 'channel_published_at', 'channel_description', 'channel_country', 'channel_video_count',
                  'channel_custom_url', 'channel_have_hidden_subscribers', 'channel_localized_title', 'channel_localized_description'], axis=1)

    df['video_published_at'] = pd.to_datetime(df['video_published_at'], errors='coerce').dt.tz_localize(None)
    df['video_trending__date'] = pd.to_datetime(df['video_trending__date'], errors='coerce').dt.tz_localize(None)

    df = df.drop_duplicates()

    df = df.sort_values(by='video_published_at', ascending=False)
    if weeks > 0:
        start_date = df['video_published_at'].iloc[0] - relativedelta(weeks=weeks)
        df = df[df['video_published_at'] >= start_date]
    elif days > 0:  
        start_date = df['video_published_at'].iloc[0] - relativedelta(days=days)
        df = df[df['video_published_at'] >= start_date]
    df.reset_index(drop=True, inplace=True)

    df['published_dayofweek'] = df['video_published_at'].dt.dayofweek
    df['published_hour'] = df['video_published_at'].dt.hour
    df['days_to_trend'] = (df['video_trending__date'] - df['video_published_at']).dt.days
    df = df[df['days_to_trend'] >= 0]

    df['video_category_id'] = df['video_category_id'].str.replace(' ', '_')

    df['video_title_length'] = df['video_title'].str.split().str.len()
    df['video_tag_count'] = df['video_tags'].str.split('|').str.len()
    df['video_tag_count'] = df['video_tag_count'].fillna(0)
    df = df.drop(['video_trending__date', 'video_tags'], axis=1)
    missing_values = ['nan', 'NaN', 'null', 'None', '', 'NULL', 'N/A', 'na']
    df.replace(missing_values, np.nan, inplace=True)
    df = df.dropna()

    session = create_retry_session()

    if threads == 0:
        max_workers = None
    else:
        max_workers = threads
    
    if detect:
        df = thumbnail_parallel_detect(df, size, max_workers)
    
    if stats:
        df = thumbnails_parallel_stats(df, max_workers)

    if embed: 
        df = thumbnail_parallel_embeddings(df, max_workers)
    df = df.drop(['video_default_thumbnail'], axis=1)

    if translate:
        df = titles_parallel_translate(df, max_workers)

    durations = df['video_duration'].fillna('').astype(str).tolist()
    with ThreadPoolExecutor(max_workers=max_workers) as executor: 
        duration_secs = list(tqdm(executor.map(convert_duration, durations), total=len(durations), desc="Converting durations"))
    df['video_duration'] = duration_secs

    df = df.sort_values(by='video_published_at')

    train_end = int(len(df) * 0.7)
    val_end = int(len(df) * 0.85)

    df_train = df.iloc[:train_end]
    df_val = df.iloc[train_end:val_end]
    df_test = df.iloc[val_end:]

    df_train, df_val, df_test, thumbnail_pca = reduce_thumbnail_embeddings_pca(df_train, df_val, df_test)

    if vectorize:
        df_train, df_val, df_test, title_vectorizer, title_encoder = titles_parallel_vectorize(df_train, df_val, df_test, stop_words, max_workers)

    if stats:
        columns_to_scale = ['thumbnail_brightness', 'thumbnail_contrast', 'thumbnail_saturation']
        stats_scaler = MinMaxScaler()    
        df_train[columns_to_scale] = stats_scaler.fit_transform(df_train[columns_to_scale])
        df_val[columns_to_scale] = stats_scaler.transform(df_val[columns_to_scale])
        df_test[columns_to_scale] = stats_scaler.transform(df_test[columns_to_scale])
    
    df_train.replace(missing_values, np.nan, inplace=True)
    df_val.replace(missing_values, np.nan, inplace=True)
    df_test.replace(missing_values, np.nan, inplace=True)
    df_train = df_train.dropna()
    df_val = df_val.dropna()
    df_test = df_test.dropna()
    
    
    df_train = df_train.drop(['video_title', 'video_title_language', 'video_title_clean', 'video_title_translated'], axis=1)
    df_val = df_val.drop(['video_title', 'video_title_language', 'video_title_clean', 'video_title_translated'], axis=1)
    df_test = df_test.drop(['video_title', 'video_title_language', 'video_title_clean', 'video_title_translated'], axis=1)

    df_train, df_val, df_test, language_pca = reduce_language_pca(df_train, df_val, df_test)

    df_train, df_val, df_test, category_encoder, category_pca = process_video_category(df_train, df_val, df_test)

    df_train.reset_index(drop=True, inplace=True)
    df_val.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)

    df_train.to_csv(PROCESSED_DATA_DIR / 'train_dataset.csv', index=False)
    df_val.to_csv(PROCESSED_DATA_DIR / 'val_dataset.csv', index=False)
    df_test.to_csv(PROCESSED_DATA_DIR / 'test_dataset.csv', index=False)    

    joblib.dump(stats_scaler, MODELS_DIR / 'stats_scaler.pkl')
    joblib.dump(title_vectorizer, MODELS_DIR / 'title_vectorizer.pkl')
    joblib.dump(title_encoder, MODELS_DIR / 'title_encoder.pkl')    
    joblib.dump(category_encoder, MODELS_DIR / 'category_encoder.pkl')    
    joblib.dump(category_pca, MODELS_DIR / 'category_pca.pkl')
    joblib.dump(thumbnail_pca, MODELS_DIR / 'thumbnail_pca.pkl')
    joblib.dump(language_pca, MODELS_DIR / 'language_pca.pkl')    

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

    title = title.lower()
    title = emoji.replace_emoji(title, replace='')
    title = re.sub(r'http\S+|www\S+|https\S+', '', title)
    title = re.sub(r'[^a-zA-Z0-9\s]', '', title)
    title = re.sub(r'\s+', ' ', title).strip()

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
        return 'unknown', title

    if lang == 'en':
        return lang, title

    try:
        translated = GoogleTranslator(source='auto', target='en').translate(title)
        return lang, translated
    except:
        return lang, title

# ---------------------------------------------

def titles_parallel_translate(df, stop_words, max_workers):
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

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        clean_titles = list(tqdm(executor.map(clean_title, titles), total=len(titles), desc="Cleaning titles"))

    languages, translations = [], []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = executor.map(detect_and_translate, clean_titles)
        for lang, translated in tqdm(futures, total=len(clean_titles), desc="Detecting & translating"):
            languages.append(lang)
            translations.append(translated)

    df = df.copy()
    df['video_title_clean'] = clean_titles
    df['video_title_language'] = languages
    df['video_title_translated'] = translations

    mask_failed_translation = (df['video_title_language'] != 'en') & (df['video_title_translated'] == df['video_title_clean'])
    df = df.loc[~mask_failed_translation].reset_index(drop=True)

    return df

# ---------------------------------------------

def titles_parallel_vectorize(df_train, df_val, df_test, stop_words, max_workers, max_features=100):
    """
    Cleans, detects language, translates, vectorizes, and encodes video titles in the provided training, validation, and test DataFrames.

    Args:
        df_train (pd.DataFrame): Training set with a 'video_title' column.
        df_val (pd.DataFrame): Validation set with a 'video_title' column.
        df_test (pd.DataFrame): Test set with a 'video_title' column.
        max_workers (int): Number of threads to use for parallel processing.
        max_features (int): Maximum number of features for TF-IDF vectorization (default: 100).

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, TfidfVectorizer, OneHotEncoder]:
            - Transformed training, validation, and test DataFrames
            - The fitted TF-IDF vectorizer
            - The fitted one-hot encoder
    """

    df_train = titles_parallel_translate(df_train, max_workers)
    df_val = titles_parallel_translate(df_val, max_workers)
    df_test = titles_parallel_translate(df_test, max_workers)

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words='english',
        ngram_range=(1, 2),
        sublinear_tf=True,
        max_df=0.8,
        min_df=2,
        norm='l2'
    )

    train_tfidf = vectorizer.fit_transform(df_train['video_title_translated'])
    val_tfidf = vectorizer.transform(df_val['video_title_translated'])
    test_tfidf = vectorizer.transform(df_test['video_title_translated'])

    tfidf_cols = vectorizer.get_feature_names_out()

    df_train_tfidf = pd.DataFrame(train_tfidf.toarray(), columns=tfidf_cols, index=df_train.index)
    df_val_tfidf = pd.DataFrame(val_tfidf.toarray(), columns=tfidf_cols, index=df_val.index)
    df_test_tfidf = pd.DataFrame(test_tfidf.toarray(), columns=tfidf_cols, index=df_test.index)

    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    lang_col = ['video_title_language']

    train_encoded = encoder.fit_transform(df_train[lang_col])
    val_encoded = encoder.transform(df_val[lang_col])
    test_encoded = encoder.transform(df_test[lang_col])

    encoded_cols = encoder.get_feature_names_out(lang_col)

    df_train_encoded = pd.DataFrame(train_encoded, columns=encoded_cols, index=df_train.index)
    df_val_encoded = pd.DataFrame(val_encoded, columns=encoded_cols, index=df_val.index)
    df_test_encoded = pd.DataFrame(test_encoded, columns=encoded_cols, index=df_test.index)

    df_train = pd.concat([df_train.reset_index(drop=True), df_train_tfidf.reset_index(drop=True), df_train_encoded.reset_index(drop=True)], axis=1)
    df_val = pd.concat([df_val.reset_index(drop=True), df_val_tfidf.reset_index(drop=True), df_val_encoded.reset_index(drop=True)], axis=1)
    df_test = pd.concat([df_test.reset_index(drop=True), df_test_tfidf.reset_index(drop=True), df_test_encoded.reset_index(drop=True)], axis=1)

    return df_train, df_val, df_test, vectorizer, encoder

# ---------------------------------------------

def reduce_language_pca(df_train, df_val, df_test, pca_variance_target=0.7, pca_max_components=10):
    """
    Applies PCA to reduce the dimensionality of video_title_language_ .

    Args:
        df_train (pd.DataFrame): Training set with language (columns starting with 'video_title_language_').
        df_val (pd.DataFrame): Validation set.
        df_test (pd.DataFrame): Test set.
        pca_variance_target (float, optional): Target cumulative explained variance for PCA. Default is 0.7.
        pca_max_components (int, optional): Maximum number of PCA components to retain. Default is 10.

    Returns:
        tuple:
            - df_train (pd.DataFrame): Train set with PCA-reduced.
            - df_val (pd.DataFrame): Val set with PCA-reduced.
            - df_test (pd.DataFrame): Test set with PCA-reduced.
            - pca (PCA): The fitted PCA model.
    """

    lang_cols = [col for col in df_train.columns if str(col).startswith('video_title_language_')]

    df_train = df_train.dropna(subset=lang_cols)
    df_val = df_val.dropna(subset=lang_cols)
    df_test = df_test.dropna(subset=lang_cols)

    X_train = df_train[lang_cols].values
    X_val = df_val[lang_cols].values
    X_test = df_test[lang_cols].values

    cumulative = np.cumsum(PCA().fit(X_train).explained_variance_ratio_)
    n_components = np.argmax(cumulative >= pca_variance_target) + 1
    n_components = min(pca_max_components, n_components)

    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)
    X_test_pca = pca.transform(X_test)

    pca_cols = [f'lang_pca_{i}' for i in range(n_components)]

    df_train_pca = pd.DataFrame(X_train_pca, columns=pca_cols, index=df_train.index)
    df_val_pca = pd.DataFrame(X_val_pca, columns=pca_cols, index=df_val.index)
    df_test_pca = pd.DataFrame(X_test_pca, columns=pca_cols, index=df_test.index)

    df_train = pd.concat([df_train.drop(columns=lang_cols), df_train_pca], axis=1)
    df_val = pd.concat([df_val.drop(columns=lang_cols), df_val_pca], axis=1)
    df_test = pd.concat([df_test.drop(columns=lang_cols), df_test_pca], axis=1)

    return df_train, df_val, df_test, pca

# ---------------------------------------------

def process_video_category(df_train, df_val, df_test, threshold=0.1, use_pca=True, pca_variance_target=0.7, pca_max_components=20):
    """
    Processes the 'video_category_id' categorical feature by applying one-hot encoding, filtering infrequent categories, and optionally reducing dimensionality using PCA.

    Args:
        df_train (pd.DataFrame): Training dataset containing the 'video_category_id' column.
        df_val (pd.DataFrame): Validation dataset.
        df_test (pd.DataFrame): Test dataset.
        threshold (float, optional): Filters dummy columns whose mean frequency in training is < threshold or > 1 - threshold. Defaults to 0.1.
        use_pca (bool, optional): Whether to apply PCA to the encoded category columns. Defaults to True.
        pca_variance_target (float, optional): Minimum cumulative explained variance to retain in PCA. Defaults to 0.7.
        pca_max_components (int, optional): Maximum number of PCA components to retain. Defaults to 20.

    Returns:
        tuple:
            - df_train (pd.DataFrame): Transformed training DataFrame.
            - df_val (pd.DataFrame): Transformed validation DataFrame.
            - df_test (pd.DataFrame): Transformed test DataFrame.
            - encoder (OneHotEncoder): The fitted one-hot encoder.
            - pca (PCA): The fitted pca in case use_pca is True.
    """

    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    train_encoded = encoder.fit_transform(df_train[['video_category_id']].astype(str))
    val_encoded = encoder.transform(df_val[['video_category_id']].astype(str))
    test_encoded = encoder.transform(df_test[['video_category_id']].astype(str))

    dummy_cols = encoder.get_feature_names_out(['video_category_id'])

    df_train_encoded = pd.DataFrame(train_encoded, columns=dummy_cols, index=df_train.index).astype(int)
    df_val_encoded = pd.DataFrame(val_encoded, columns=dummy_cols, index=df_val.index).astype(int)
    df_test_encoded = pd.DataFrame(test_encoded, columns=dummy_cols, index=df_test.index).astype(int)

    keep_cols = [col for col in dummy_cols if threshold < df_train_encoded[col].mean() < 1 - threshold]

    df_train = pd.concat([df_train.drop(columns=['video_category_id']), df_train_encoded[keep_cols]], axis=1)
    df_val = pd.concat([df_val.drop(columns=['video_category_id']), df_val_encoded[keep_cols]], axis=1)
    df_test = pd.concat([df_test.drop(columns=['video_category_id']), df_test_encoded[keep_cols]], axis=1)

    df_train = df_train.dropna()
    df_val = df_val.dropna()
    df_test = df_test.dropna()

    if use_pca:
        cumulative = np.cumsum(PCA().fit(df_train_encoded).explained_variance_ratio_)
        n_components = np.argmax(cumulative >= pca_variance_target) + 1
        n_components = min(pca_max_components, n_components)

        pca = PCA(n_components=n_components)
        df_train_pca = pd.DataFrame(pca.fit_transform(df_train_encoded), index=df_train.index)
        df_val_pca = pd.DataFrame(pca.transform(df_val_encoded), index=df_val.index)
        df_test_pca = pd.DataFrame(pca.transform(df_test_encoded), index=df_test.index)

        df_train_pca.columns = [f'video_category_pca_{i}' for i in range(n_components)]
        df_val_pca.columns = df_train_pca.columns
        df_test_pca.columns = df_train_pca.columns

        df_train = pd.concat([df_train.drop(columns=keep_cols), df_train_pca], axis=1)
        df_val = pd.concat([df_val.drop(columns=keep_cols), df_val_pca], axis=1)
        df_test = pd.concat([df_test.drop(columns=keep_cols), df_test_pca], axis=1)

        return df_train, df_val, df_test, encoder, pca
    else:
        return df_train, df_val, df_test, encoder

# ---------------------------------------------

def create_retry_session():
    """
    Creates and returns a requests Session object with automatic retry logic.

    The session is configured to retry failed HTTP requests up to 3 times with an exponential backoff (1s, 2s, 4s) 
    for specific server error codes (500, 502, 503, 504). This helps improve reliability when dealing with 
    temporary server issues or intermittent network errors.
    
    Returns:
        requests.Session: A session object with retry capabilities.
    """

    session = requests.Session()
    retries = Retry(
        total=2,
        backoff_factor=1,
        status_forcelist=[500, 502, 503, 504],
        raise_on_status=False
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

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
    detections_df = detections_df.iloc[:, :-1]
    df = pd.concat([df, detections_df], axis=1)

    return df

# ---------------------------------------------

def thumbnail_stats(thumbnail_url, idx, pbar):
    """
    This function computes the brightness, contrast, and saturation of a thumbnail image given its URL. It fetches the image, 
    calculates the required statistics, and returns the results.

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
    except Exception as e:
        pbar.update(1)

        return idx, [np.nan, np.nan, np.nan]

# ---------------------------------------------

def thumbnails_parallel_stats(df, max_workers):
    """
    Computes the brightness, contrast, and saturation statistics for each thumbnail in the provided DataFrame. The statistics 
    are calculated in parallel using a ThreadPoolExecutor for faster processing.

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

    df_stats = pd.DataFrame(stats_array, columns=['thumbnail_brightness', 'thumbnail_contrast', 'thumbnail_saturation'])

    df = pd.concat([df.reset_index(drop=True), df_stats.reset_index(drop=True)], axis=1)
    return df

# ---------------------------------------------

def embedding_thumbnail(thumbnail_url, idx, transform, model, pbar):
    """
    This function processes a thumbnail image from a given URL, extracts its features using a pre-trained model, and returns 
    the index of the image along with the extracted feature vector.

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
        features = np.full((1280,), np.nan) 
    pbar.update(1)
    
    return idx, features

# ---------------------------------------------

def thumbnail_parallel_embeddings(df, max_workers):
    """
    This function extracts embeddings from the thumbnails of videos in the given dataframe. It uses a pre-trained MobileNetV2 
    model to generate embeddings for each thumbnail URL. 

    Args:
    df (pd.DataFrame): DataFrame containing a column 'video_default_thumbnail' with URLs to video thumbnails.
    max_workers (int): Maximum number of workers for parallel processing. 

    Returns:
    pd.DataFrame: The original DataFrame with additional columns containing the reduced thumbnail embeddings.
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
    
    embed_cols = [f'thumb_emb_{i}' for i in range(embeddings_array.shape[1])]  
    embeddings_df = pd.DataFrame(embeddings_array, columns=embed_cols)
    df = pd.concat([df.reset_index(drop=True), embeddings_df], axis=1)
    
    return df

# ---------------------------------------------

def reduce_thumbnail_embeddings_pca(df_train, df_val, df_test, pca_variance_target=0.7, pca_max_components=100):
    """
    Applies PCA to reduce the dimensionality of thumbnail embeddings.

    Args:
        df_train (pd.DataFrame): Training set with thumbnail embeddings (columns starting with 'thumb_emb_').
        df_val (pd.DataFrame): Validation set.
        df_test (pd.DataFrame): Test set.
        pca_variance_target (float, optional): Target cumulative explained variance for PCA. Default is 0.95.
        pca_max_components (int, optional): Maximum number of PCA components to retain. Default is 100.

    Returns:
        tuple:
            - df_train (pd.DataFrame): Train set with PCA-reduced embeddings.
            - df_val (pd.DataFrame): Val set with PCA-reduced embeddings.
            - df_test (pd.DataFrame): Test set with PCA-reduced embeddings.
            - pca (PCA): The fitted PCA model.
    """

    embed_cols = [col for col in df_train.columns if str(col).startswith('thumb_emb_')]

    df_train = df_train.dropna(subset=embed_cols)
    df_val = df_val.dropna(subset=embed_cols)
    df_test = df_test.dropna(subset=embed_cols)

    X_train = df_train[embed_cols].values
    X_val = df_val[embed_cols].values
    X_test = df_test[embed_cols].values

    cumulative = np.cumsum(PCA().fit(X_train).explained_variance_ratio_)
    n_components = np.argmax(cumulative >= pca_variance_target) + 1
    n_components = min(pca_max_components, n_components)

    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)
    X_test_pca = pca.transform(X_test)

    pca_cols = [f'thumb_pca_{i}' for i in range(n_components)]

    df_train_pca = pd.DataFrame(X_train_pca, columns=pca_cols, index=df_train.index)
    df_val_pca = pd.DataFrame(X_val_pca, columns=pca_cols, index=df_val.index)
    df_test_pca = pd.DataFrame(X_test_pca, columns=pca_cols, index=df_test.index)

    df_train = pd.concat([df_train.drop(columns=embed_cols), df_train_pca], axis=1)
    df_val = pd.concat([df_val.drop(columns=embed_cols), df_val_pca], axis=1)
    df_test = pd.concat([df_test.drop(columns=embed_cols), df_test_pca], axis=1)

    return df_train, df_val, df_test, pca

# ---------------------------------------------------------------------------------------------------------------------------    

if __name__ == "__main__":
    app()