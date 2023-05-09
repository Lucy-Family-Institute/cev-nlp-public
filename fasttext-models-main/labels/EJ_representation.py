"""
This file is for a collaboration with Santiago Pardo,
where vector representations are made for etiquetas of
Homicidio
"""

import os
import fasttext
import pandas as pd
from preprocess import PreprocessPipeline
import spacy
import numpy as np

path = os.path.dirname(os.path.abspath(__file__))
labels_path = "/rep/repositories/dataset/survey/fichas_labels.csv.gz"
model_path = os.path.join(path, "..", "models", "model_interviews.bin")

# Get etiquetas from Homicidio:
chunksize = 10 ** 5
usecols = ["interview_id", "label", "group", "text"]
chunks = []
with pd.read_csv(labels_path, sep="¬", compression="gzip", usecols=usecols, chunksize=chunksize) as reader:
    for chunk in reader:
        chunk = chunk[chunk.label.str.contains("Ejecucion Extrajudicial")]
        chunks.append(chunk.drop_duplicates(subset="text"))
df = pd.concat(chunks).drop_duplicates(subset="text")
print("Ejecución extrajudicial dataframe has shape", df.shape)
print("There are", len(df.interview_id.unique()), "unique ids")

# Get rid of short texts:
df = df[df.text.apply(lambda x: True if len(x) >= 200 else False)].reset_index(drop=True)
print("After dropping short texts there are", len(df.interview_id.unique()), "unique ids")

# Clean text:
nlp = spacy.load("es_core_news_lg")
pp = PreprocessPipeline(tokenise=False, nlp=nlp)
clean_texts = pp(df.text.apply(lambda x: x.replace("\n", " ")), num_cpus=-1)

# Generate vectors:
model = fasttext.load_model(model_path)
vectors = np.array([model.get_sentence_vector(text) for text in clean_texts])
for j in range(vectors.shape[1]):
    df[f"fasttext_component_{j+1}"] = vectors[:, j]

df.to_excel("Ejecucion_extrajudicial_vec_representation.xlsx", index=False)
print("Final dataframe has shape", df.shape)

average_data = []
# Average vectors:
for interview_id, group in df.groupby("interview_id"):
    data = [interview_id]
    vector = group.iloc[:, 4:].to_numpy().mean(axis=0).tolist()
    data.extend(vector)
    average_data.append(data)

new_columns = ["interview_id"] + df.columns.values.tolist()[4:]
average_df = pd.DataFrame(data=average_data, columns=new_columns)
average_df.to_excel("homicidio_vec_representation_average.xlsx", index=False)
print("Final average dataframe has shape", average_df.shape)

