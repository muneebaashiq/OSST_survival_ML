# OSST_survival_ML

## Table of contents <!-- omit in toc -->

- [Installation](#installation)
- [Configuration](#configuration)
- [Run](#run)

## Installation

In the project directory, i.e. where the **pyproject.toml** file is located, use the following commands to set up a virtual environment and install all project dependencies:

```bash
    python3 -m venv env
    source env/bin/activate
    pip install poetry
    poetry install
    pip install osst
```

## Configuration

Make sure to configure everything needed for your experiments in the **config.yaml** file.\
Most important is the path to your input_file and the names of your events and times columns.

### Notes for updating the config.yaml file
1. The complete dataset that you have needs to separated into the train and test dataset before running the pipeline.
2. The train and test datasets needs to be excel files.
3. The path to the train dataset excel file will go in the “in_file” entry in the config.yaml file.
4. The path to the test dataset excel file will go in the “test_file” entry in the config.yaml file.
5. The time and event columns in your dataset should be either name “Time” and “Event respectively or their names must be changed in the config.yaml file.
6. The are multiple models/processes need to done to prepare the data for OSST model execution all of them can be changed if it is taking too long for the model to train.

## Run

After the config file is set up properly, you can run the pipeline using:

```bash
python3 main.py
```

Results are automatically saved after each iteration and will not be recomputed unless the meta.overwrite flag is set to True.
