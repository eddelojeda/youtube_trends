<center>
    <p>
        <img src="https://mcd.unison.mx/wp-content/themes/awaken/img/logo_mcd.png" width="100" alt="Logo MCD">
    </p>
</center>

# YouTube Trends Predictor

A data-driven project analyzing YouTube trending videos and predicting virality using machine learning. Includes data visualization, trend insights, and predictive models.

## Prior Requirements
In order to execute this project, it is necessary to have the following programs in place beforehand:

* Python 3.10.0+
* pip 25.0.1+
* Makefile (optional)

## Project Organization
<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
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
    ├── dataset.py              <- Scripts to download or generate data
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

## Clone Project
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

## Virtual Environment Creation
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

## Dependency installation

To install the necessary dependencies, you can use the commands:
```bash
make requirements
```
or the command:
```bash
pip install -r requirements.txt
```

## Make Commands

List of commands available for the Makefile:

* `make create_environment`: Create a virtual environment and print the necessary command to activate it.
* `make init`: Initializes the project and creates the necessary files.
* `make requirements`: Install the required libraries from the `requirements.txt` file.
* `make process`: Processes the data in the `/data/raw` folder and saves the results to `/data/processed`.

--------

