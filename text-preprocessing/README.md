# Text-Preprocessing module

This is a module to perform text preprocessing on the several corpora of the Comisión de la Verdad.

# Installation

The module can is pip installable (latest version is **0.0.4**):

```zsh
pip install git+https://gitlab.com/nlp-comision/text-preprocessing.git@v0.0.4
```

# Gettint Started

The module can be used very easily:

```python
from preprocess import PreprocessPipeline
import spacy

nlp = spacy.load("es_core_news_sm", disable=['parser', 'ner'])

pp = PreprocessPipeline(nlp=nlp)
print(pp("hola amiguitos! qué tal todo?"))
print(pp(["hola amiguitos! qué tal todo?"]*1000, num_cpus=3))
```

Note that `num_cpus` will set the number of cores that will distribute computation on the given list of texts to be preprocessed. Check the documentation of `PreprocessPipeline`, as there are plenty of options to be set.

# Troubleshooting

When doing parallel preprocessing, it might be an issue with `huggingface` based `spaCy` models that transformers cannot be run in parallel. Apparently, downgrading works:

```python
pip install transformers==3.0.2
```
