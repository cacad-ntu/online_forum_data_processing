# Online Forum Data Processing

This is a project for CZ4045 Natural Language Processing assignment.

This experiment consists of 3 main steps which are tokenization, pos-tagging, and further analysis. Firstly, tokenizer is used to divide sentences into several tokens. Next, pos-tagger will annotate each of the tokens based on its tags. To process information from online forums, we need to develop a specific tokenizer and pos-tagger to handle irregular tokens in the sentences. Additionally, irregular token needs to be defined and annotated manually for training data. Lastly, further analysis will be done to develop the real-world application by using the tokenizer and pos-tagger. The applications developed include negative sentence analyzer, semantic sentence analyzer and exception handling sentence analyzer. The objective of developing those applications is to observe behaviours of several NLP techniques. The NLP techniques used are Recursive Neural Network (RNN), Support Vector Machine (SVM), Naive Bayes, and regex.

## Introduction

In the recent years, a lot of research has been done to find out how to analyze and process big data. One of the fields is Natural Language Processing. Natural Language Processing is concerned about programming the computer to understand certain language and to enable the interaction between human and computer. Nowadays, Natural Language Processing is used to analyze/understand the language written or spoken in daily conversation. However, in the online forums, the discussed items may not only contain human language, but also contain code snippets, special terms, etc. In Natural Language Processing, those entities must be treated as irregular tokens. To analyze sentences in online forums such as Stack Overflow, additional steps are required on top of the regular NLP. These additional steps will be the main methods to handle the irregular tokens. So, the following section discusses one of the methods to solve this problem.

## Getting Started

### Requirement

- Python 2.7 (with pip)

### Installing

- Clone this repo.
- Move to the root directory of this project.
- Run `pip install -r requirement.txt`.
- Get [raw_data.xml](https://drive.google.com/file/d/0B0QpPMrU8F0ATGFTZmJiU0VUaU0/view?usp=sharing) put under [`data/`](data/) directory OR do the following steps (on UNIX platform):
  - Download raw xml file from [here](https://archive.org/details/stackexchange).
  - put under `data/` dicrectory and rename as `raw_data.xml`.
  - Run `split --bytes=200M raw_data.xml`.
- Execute `python start.py`.

### Dataset Collection

To collect data from [raw_data.xml](https://drive.google.com/file/d/0B0QpPMrU8F0ATGFTZmJiU0VUaU0/view?usp=sharing) do the following.

- Move the current directory to `dataset_collection`.
- Execute `python collector_data.py`.

### Dataset Analysis and Annotation

#### Stemming

To stem the raw data and get the count before and after stemming do the following.

- Move the current directory to `dataset_analysis`.
- To stem data execute `python stemming.py`.
- The stem result can be found at [result_stemmed.json](data/result_stemmed.json).
- The original word count can be found at [result_word_count.json](data/result_word_count.json).

#### POS Tagging

- Move the current directory to `dataset_analysis`.
- To POS tag data execute `python pos_tagging.py`.
- The POS tag result can be found at [pos_tag.json](data/pos_tag.json).

#### Data Annotation

To annotate data we need to split the data for easier annotation.

- Move the current directory to `dataset_analysis`.
- To split data execute `python split_data.py`.
- result can be found at `data/` directory.
- You can start manual annotation from that files.
- The final annotated file for tokenizer is at [train_data.json](`data/train_data.json) and for application at [data_class.json](data/data_class.json).

### Tokenizer

Tokenizer will be used in the further analysis and application

### Further Analysis

#### Irregular Token Tokenizer

To tokenize the irregular token. We need to use the regex or crf tokenizer by do the following:

- Move the current directory to `further_analysis`.
- Execute `python count_tokenizer.py`.
- The result for regex tokenizer can found at [result_new_token_regex_count.json](data/result_new_token_regex_count.json).
- The result for CRF tokenizer can found at [result_new_token_crf_count.json](data/result_new_token_crf_count.json).

#### Normal POS Tagging with Regex Tokenizer

To test our tokenizer we can try to apply POS tag with regex tokenizer.

- Move the current directory to `further_analysis`.
- Execute `python normal_pos_tagging.py`.
- The result can be found at [pos_tag_normal.json](data/pos_tag_normal.json).

#### CRF POS Tagging with Regex Tokenizer

To try the CRF POS tagging can do the following:

- Move the current directory to `further_analysis`.
- Execute `python crf_pos_tag.py`.
- The model will be found at the same directory (*.crfsuite).
- The result will be printed on terminal.

### Application

The application default settings are

- For error sentence application are using tuned SVM.
- For semantic analysis application is using tuned Naive Bayes.
- For negation expression application are using tuned SVM.

To change the application setting can directly change [application.py](application.py) To run the application using the default setting do the followings.
- Execute `python application.py`.
- Choose on of the following application
  1. Error sentence application
  1. Negative expression application
  1. Semantic Analysis application
  1. Negative application using regex
- Enter the sentence you want to classify.
- Choose `5. Exit` to exit from the application

Additionally, you can try the application without installing the requirements [here](http://128.199.159.74:8888/?token=117369b26b2baa3a96c5f1710c413583517d54793ee189b6).

## Report

The report of this experiment can be found [here](report.pdf).