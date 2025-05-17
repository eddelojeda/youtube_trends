<center>
    <p>
        <img src="https://mcd.unison.mx/wp-content/themes/awaken/img/logo_mcd.png" width="100" alt="Logo MCD">
    </p>
</center>

# 🚀 YouTube Trends Predictor 

A data-driven project analyzing YouTube trending videos and predicting virality using machine learning. Includes data visualization, trend insights, and predictive models.

## 🔧 Prior Requirements
In order to execute this project, it is necessary to have the following programs in place beforehand:

* Python 3.10.0+
* pip 25.0.1+
* Makefile (optional)

## 📂 Project Organization
<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         youtube_trends and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── youtube_trends   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes youtube_trends a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download and preprocess the data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

## 📥 Clone Project
To clone the project to your computer, run the following command line:

```bash
git clone https://github.com/eddelojeda/youtube_trends.git
```

To validate that the project has been cloned correctly, run the following commands to verify that you have the latest version of the project:

```bash
cd youtube_trends
git status
```

If you have the `Makefile` program, to give your project the desired structure, run the following command, which will create all the necessary folders to keep all your information organized:

```bash
make init
```

## 🐍 Virtual Environment Creation
To run the project, the native Python `venv` option was used. First, navigate to your project folder.

One option is to use the `make` command in our project folder to automatically create it:

```bash
make create_environment
```

If you want to do it manually, you can create it with the following commands. It's recommended to create this environment in the project folder:

```bash
cd C:\path\to\project
python -m venv <Virtual_Environment_Name>
```

To activate the environment, use the following command:

```bash
# Windows
.\Virtual_Environment_Name\Scripts\activate

#Linux
source Virtual_Environment_Name/bin/activate
```

To deactivate the virtual environment, use the command:

```bash
deactivate
```

## 📦 Dependency installation

To install the necessary dependencies, you can use the commands:
```bash
make requirements
```
or the command:
```bash
pip install -r requirements.txt
```

## ⚙️ Make Commands

List of commands available for the Makefile:

* `make create_environment`: Create a virtual environment and print the necessary command to activate it.
* `make init`: Initializes the project and creates the necessary files.
* `make requirements`: Install the required libraries from the `requirements.txt` file.
* `make process`: Processes the data in the `/data/raw` folder and saves the results to `/data/processed`.

## ▶️ How to use

To download and preprocess the data, you need to run the dataset.py script. This script supports several parameters to control the data processing pipeline. By default, the parameters are as follows:

```python
input_path: Path = RAW_DATA_DIR / "dataset.csv"
output_train_path: Path = PROCESSED_DATA_DIR / "train_dataset.csv"
output_val_path: Path = PROCESSED_DATA_DIR / "val_dataset.csv"
output_test_path: Path = PROCESSED_DATA_DIR / "test_dataset.csv"

redownload: bool = typer.Option(False, "--redownload", "-r", help="Download raw dataset. Default value: False.")
vectorize: bool = typer.Option(False, "--vectorize", "-v", help="Vectorize and detect language of the video title. Default value: False.")
translate: bool = typer.Option(False, "--translate", "-t", help="Translate the video titles to English. Default value: False.")
detect: bool = typer.Option(False, "--detect", "-d", help="Detect objects in thumbnail images. Default value: False.")
stats: bool = typer.Option(False, "--stats", "-s", help="Compute brightness, contrast, and saturation stats in thumbnails. Default value: False.")
embed: bool = typer.Option(False, "--embed", "-e", help="Extract embeddings from thumbnails. Default value: False.")
size: str = typer.Option("n", "--size", help="Specify YOLOv5 version for processing (n, s, m, l, x). Default: 'n'.")
weeks: int = typer.Option(0, "--weeks", help="Number of weeks to use from the raw dataset. Default: 0 (full dataset).")
days: int = typer.Option(0, "--days", help="Number of days to use from the raw dataset. Default: 0 (full dataset).")
threads: int = typer.Option(0, "--threads", help="Number of threads for parallel processing. Default: 0 (auto).")
```

Example usage:

```
python youtube_trends/dataset.py -r -v -t -d --s -e --size=n
```

This will:
- ⬇️ Download the raw dataset (if not already downloaded or if --redownload is set),
- 📝 Vectorize and detect the language of video titles,
- 🔤 Translate titles to English,
- 🔍 Detect objects in thumbnails,
- 🎨 Compute thumbnail stats (brightness, contrast, saturation),
- 🧠 Extract thumbnail embeddings,
- 🔧 Select YOLOv5n model version to perform object detection in thumbnails,
- 📅 Process the full dataset (no limits on weeks/days),
- ⚙️ Automatically select number of threads for parallel processing.

--------

