# mlops_pirate
MLOps "producionalizing" an ipynb

## day 1

this project is based on https://github.com/alexeygrigorev/ml-engineering-contsructor-workshop

### how to install UV
just run `curl -LsSf https://astral.sh/uv/install.sh | sh`

### day 1 steps

- create a folder `day_1`
- change directory into it via `cd day_1`
- run `uv init --python 3.10` to initialize a uv project
- run `uv sync`

### download the notebook:

- `mkdir notebooks`
- `cd notebooks`
- `wget "https://raw.githubusercontent.com/alexeygrigorev/ml-engineering-contsructor-workshop/refs/heads/main/01-train/duration-prediction-starter.ipynb"`
- `cd ..`

### install dependencies
- `uv add scikit-learn==1.2.2 pandas pyarrow`
- fix issue with numpy via `uv add numpy==1.26.4`
- install jupyter `uv add --dev jupyter seaborn`

### launch jupyter notebook
`uv run jupyter notebook`

### convert notebook into a script
- `uv run jupyter nbconvert --to=script notebooks/duration-prediction-starter.ipynb`
- create a new folder `duration_prediction` and move the file `duration-prediction-starter.py` into that folder and rename it into `train.py`

### lets make the train.py script nice
- install python vs code extension
- make vscode use correct python version:
    - click bottom right
    - enter interpreter path
    - find
    - `/workspaces/2025-07-mle-workshop/day_1/.venv/bin/python`

- remove # lines from script
- move imports all to top
- remove top level statements
- improve code
- we can run it via `uv run python duration_prediction/train.py`
- parametrize the train function
- use argparse: now we can call it via `uv run python duration_prediction/train.py --train-date 2022-01 --val-date 2022-02 --model-save-path out_file.np`
- add docstrings to functions
- logging: `uv add loguru`
- split out argparse into main.py and make it a module. now we call our code via `uv run python -m duration_prediction.main --train-date 2022-01 --val-date 2022-02 --model-save-path out_file.np`


### create a makefile
now we can run our training via `make train`

### tests
- `uv add pytest`
- `mkdir tests`
- create a `__init__.py` file inside of `tests`
- create a file `test_train.py` inside of `tests` (it has to start with `test_`)
- run tests via `uv run pytest` or `make test`