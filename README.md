# TEI
TEI - TimeBankPT Event Identification
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FORMAS/TEI/blob/main/notebook/colab-tei.ipynb)

[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)](https://hub.docker.com/r/andersonsacramento/tei)


# DESCRIPTION

TEI is an event trigger identifier system for sentences in the Portuguese language. It locates the event trigger terms in a sentence. The model was trained on the TimeBankPT (COSTA; BRANCO,2012) corpus.


The system outputs the identified events  in the following Json format:
```json
[
    {
        "text": "Vazamentos",
        "start": 0,
        "end": 10
    },
    {
        "text": "expõem",
        "start": 20,
        "end": 26
    },
    {
        "text": "diz",
        "start": 62,
        "end": 65
    }
]
```

# Local Execution

## Prerequisites

1. Download and place the BERTimbau Base (SOUZA; NOGUEIRA;LOTUFO, 2020) model and vocabulary file:
    ```bash
    $ wget https://neuralmind-ai.s3.us-east-2.amazonaws.com/nlp/bert-base-portuguese-cased/bert-base-portuguese-cased_tensorflow_checkpoint.zip
	```
	```bash
	$ wget https://neuralmind-ai.s3.us-east-2.amazonaws.com/nlp/bert-base-portuguese-cased/vocab.txt
	```
	Then unzip and place it in the the models directory as follows:
	```
	├──models
	|      └── BERTimbau
	|               └── bert_config.json
	|               └── bert_model.ckpt.data-00000-of-00001
	|               └── bert_model.ckpt.index
	|               └── bert_model.ckpt.meta
	|               └── vocab.txt
	|
	|...
	```
2. Install the packages.
   ```bash
   $ pip install -r requirements.txt
   ```
# OPTIONS
    -h, --help                           Print this help text and exit
	--sentence  SENTENCE                 Sentence string to identify events from
	--dir   INPUT-DIR OUTPUT-DIR         Identify events from files of input directory
		                             (one sentence per line) and write output json
					     files on output directory.



## EVENT IDENTIFICATION FROM A DIRECTORY OF FILES
The text files in the input directory are expected to have the format:

    * all text files end with the extension .txt
    * sentences are separated by newlines
	
```bash
$ python3 src/tei.py --dir /tmp/input-dir /tmp/output-dir
```
## EVENT IDENTIFICATION FROM A SENTENCE

```bash
$ python3 src/tei.py --sentence 'Vazamentos de dados expõem senhas de funcionários do governo, diz relatório.'
```
## How to cite this work

Peer-reviewed accepted paper:

* Sacramento, A., Souza, M.: Joint Event Extraction with Contextualized Word Embeddings for the Portuguese Language. In: 10th Brazilian Conference on Intelligent System, BRACIS, São Paulo, Brazil, from November 29 to December 3, 2021.