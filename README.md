# Electronics Research Institute Sharif University of Technology

## Automatic Speech Recognition Project

## Table of Contents
- [Dataset](#Dataset)
- [Quick-Installation-Usage](#Quick-Installation-Usage)

# Dataset 

**Note: For each specific dataset, it is necessary to create train_split.tsv, test_split.tsv, and valid_split.tsv based on the path of your dataset in this [**folder**](./dataset).**

This [**folder**](./dataset) contains the project dataset for Automatic Speech Recognition.
<div align="justify"> In this project we used CommonVoice-V16. This dataset is an open-source dataset developed by Mozilla. The primary goal of collecting data for this dataset is to gather audio samples from various individuals worldwide to use these samples for developing and improving speech recognition and synthesis technologies. Version 16.1 of this dataset includes the latest data and features collected so far.

## Dataset Overview

This dataset contains 364 hours of Persian audio along with corresponding text transcriptions. The audio files are stored in WAV format. Despite the claims made by the publishers of this dataset, there are numerous errors within the dataset that need correction before use. To address this, a data cleaning process has been implemented, which includes the following steps:

1. **Normalization of Text Labels:** Using the `hazm` library, which is the equivalent of `nltk` for the Persian language.
2. **Replacement of Incorrectly Spaced Words:** Correcting words that are incorrectly joined or separated in the text labels.
3. **Substitution of Similar Letters with Different Unicode:** Replacing letters that look similar but have different Unicode values, typically those imported from Arabic into Persian.
4. **Removal of Abbreviations**
5. **Conversion of Numbers to Words:** Converting numerical digits (e.g., 4) into their word form (e.g., "چهار").

By following these steps, the dataset can be refined for more accurate and reliable usage in speech technology applications.

## Data collection procedure

In this project, the CommonVoice Persian version 16 database has been used to build a proper ASR database in Persian language.
CommonVoice is an open source project started by Mozilla to collect speech data, where people can speak sentences.

```bibtex
@article{author2024commonvoice,
title={Common Voice: A Large-Scale Open-Source Database for Multilingual Speech Recognition},
author={Author, First and Author, Second},
journal={Journal of Speech and Language Technology},
volume={60},
number={2},
pages={100--120},
year={2024},
publisher={TechPublishers}
}}
```

# Quick-Installation-Usage
## Install

Once you have created your Python environment (Python 3.10.14) you can simply type:

```
pip install -r requirements.txt
```

## Usage
### Train, Evaluate, Test and Deploy Model
#### 1. Run the following code with the desired settings to train the model: ####

```bash                  
python train.py 
```
#### For example: ####

```bash                  
python train.py --version [ENTER_A_NAME_FOR_YOUR_TRAINING]
                --bs [ENTER_TRAINING_BATCHSIZE]
                --lr [ENTER_YOUR_LEARNING_RATE]
                --maxsteps [ENTER_MAXIMUM_NUMBER_OF_TRAINING_STEPS] 
                --evalsteps [ENTER_NUMBER_OF_STEPS_FOR_EACH_MODEL_EVALUATION]
                --savesteps [ENTER_NUMBER_OF_STEPS_TO_SAVE_MODEL]
                --train_csv [ENTER_PATH_TO_YOUR_TRAIN_TSV_FILE]
                --test_csv [ENTER_PATH_TO_YOUR_TEST_TSV_FILE]
```
#### 2. Run the following code with the desired settings to inference the model: ####

#### For single audio: ####

```bash
python inference.py --mode 'sample'
                    --path_audio [PATH_TO_YOUR_AUDIO_FILE]
                    --device [DEIVCE_TO_RUN_THE_CODE]
                    --ckp [PATH_TO_YOUR_MODEL_CHECKPOINT]
                    --save_path [PATH_TO_SAVE_RESULTS]
```

#### For csv file: ####

```bash
python inference.py --mode 'csv'
                    --path_csv [PATH_TO_YOUR_CSV_FILE]
                    --device [DEIVCE_TO_RUN_THE_CODE]
                    --ckp [PATH_TO_YOUR_MODEL_CHECKPOINT]
                    --save_path [PATH_TO_SAVE_RESULTS]
```
