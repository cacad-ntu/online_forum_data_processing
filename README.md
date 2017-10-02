# Online Forum Data Processing
This is a project for CZ4045 Natural Language Processing assignment.

## Introduction

Apa aja yang ada di repo ini

## Getting Started

### Requirement

- Python 2.7 or above 
- PIP

### Installing

1. Clone this repo.
1. Move to the root directory of this project.
1. Run `pip install -r requirement.txt`.
1. Download raw xml file from [here](https://archive.org/details/stackexchange).
1. Run `python start.py`

## TODO

### Code

- regex tokenizer
- CRF tokenizer
- Manual Annotation

### Report

- Dataset Collection
    - Explain our xml parser
    - Snippet [dataset_collection](./dataset_collection/) code
    - Justify raw data satisfy the three conditions
    - Distribution of raw thread (1 answer, 2 answers, more than 2 answers)
        - Show as statistics
- Dataset Analysis and Annotation
    - Stemming
        - Data
            - Count of original word
            - Count of stemmed word and their respective original word(s)
        - Identify the top-20 most frequent words (excluding the [stop words](https://www.ranks.nl/stopwords))
        - Discuss the results
            - specific based on top 20
    - POS Tagging
        - 10 sentences from the dataset
        - Discuss the tagging results
    - Token Definition and token annotation
        - Define Token (list token)
        - Annotation process (manual annotation)
- Tokenizer
    - Regex
        - Explain regex
        - Evaluate results
    - CRF
        - Data preprocessing
            - Feature selection
        - K-Folds
        - Cross Validation
        - Evaluate results
- Further Analysis
    - Irregular Tokens
        - Top-20 (not english word)
        - Sho and discuss
    - POS Tagging
        - 10 Sentences
        - POS Tagging based on our tokenizer
        - Show and discuss tagging results
- Application
    - Classifier
        - Random Forest
        - Etc.
    - Discuss the results