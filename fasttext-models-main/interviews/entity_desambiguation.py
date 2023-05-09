import os
from preprocess import PreprocessPipeline
import fasttext
import json
import pandas as pd
import umap
import umap.plot
import altair as alt
import numpy as np

path = os.path.abspath(os.path.dirname(__file__))

# Load the data:

path_interviews = "/home/vvargas/corpus/raw/04_entrevistas/all_interviews_2021_03_18_all_leves.json"
# path_interviews = "/home/vladimir/Documents/Projects/ComisionDeLaVerdad/survey-analysis/src/co-occurrence/import/input/survey_labelled_sample_500_level_4.json"
entities = []
entity_types = []
ids = []
with open(path_interviews, "r") as interviews:
    for line in interviews.readlines():
        line = json.loads(line)
        _id = line["identifier"]
        if _id in ids:
            continue
        try:
            annotations = line["labelled"]["annotation"]
        except (TypeError, KeyError):
            continue
        if annotations is None:
            continue
        ids.append(_id)
        for annotation in annotations:
            labels = annotation["label"]
            point = annotation["points"][0]
            for label in labels:
                if "Entidades" in label and "Organizaciones" in label:
                    entity = point["text"]
                    etype = ' - '.join(label.split(" - ")[1:])
                    entities.append(entity)
                    entity_types.append(etype)

df = pd.DataFrame(dict(entity=entities, type=entity_types))
counts = df.entity.value_counts()
print('Original shape', df.shape)
df = df.drop_duplicates(subset=["entity"])
df = df.dropna()
print('After drop duplicates', df.shape)
df['counts'] = [counts[ent] for ent in df.entity]

pp = PreprocessPipeline(tokenise=False, lemmatise=False, symbols=["¿", "¡", "´", "‘"])
df["clean_entity"] = pp(df.entity, num_cpus=-1)
counts = df.groupby("clean_entity")["counts"].sum()
df = df.drop_duplicates(subset=["clean_entity"])
df = df.dropna()
print('After drop duplicates with clean entities', df.shape)
df['counts'] = [counts[ent] for ent in df.clean_entity]
df["logcount"] = np.log(df.counts)
df["point_size"] = 10 + np.digitize(df.logcount.to_numpy(), [0] + np.quantile(df.logcount[df.logcount > 0], [0.25, 0.5, 0.75, 1.0]).tolist())**2 * 3

print(df.point_size.value_counts())

model = fasttext.load_model("../models/model_interviews.bin")
vectors = []
for w in df.clean_entity:
    if " " in w:
        vectors.append(model.get_sentence_vector(w))
    else:
        vectors.append(model.get_word_vector(w))

reducer = umap.UMAP(n_neighbors=10, min_dist=0.1)
embedding = reducer.fit_transform(vectors)

# FIXME: this isn't working due to Nans appearing when hovering:
# https://github.com/lmcinnes/umap/issues/351
# umap.plot.output_file('entities.html')

# p = umap.plot.interactive(reducer, labels=df.type, hover_data=df[["type", "clean_entity"]], point_size=5)
# umap.plot.show(p)

df["x"] = embedding[:, 0]
df["y"] = embedding[:, 1]

alt.data_transformers.disable_max_rows()
chart = alt.Chart(df).mark_circle().encode(
    x='x', y='y', color='type', tooltip=['clean_entity', 'point_size'], size='point_size'
).properties(
    width=800,
    height=600
).interactive()

chart.save('entities.html')