from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="preprocess",
    packages=["preprocess", "preprocess.cev"],
    version="0.0.4",
    description="Library to perform preprocessing on general texts, and also on specific texts for Comisión de la Verdad",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Comisión para el Esclarecimiento de la Verdad, la Convivencia y la no Repetición",
    author_email="{vladimir.vargas}@comisiondelaverdad.co",
    license="GNUv3",
    install_requires=[
        "typeguard==2.11.1",
        "nltk==3.5",
        "spacy==3.0.3",
        "tqdm>=4.42.0",
        "transformers==3.0.2",
    ],
    python_requires=">=3.6",
)
