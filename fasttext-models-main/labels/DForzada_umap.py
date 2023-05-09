import pandas as pd
import umap
import umap.plot
from sklearn.preprocessing import LabelEncoder

df = pd.read_excel("d_forzada_vec_representation.xlsx")
vectors = df.iloc[:, 4:].to_numpy()
le = LabelEncoder()
classes = le.fit_transform(df.label.values)

reducer = umap.UMAP(n_neighbors=10, min_dist=0.1)
embedding = reducer.fit_transform(vectors, y=classes)

umap.plot.output_file('d_forzada_map.html')
p = umap.plot.interactive(reducer, labels=df.label, hover_data=df[["label", "interview_id", "text"]], point_size=5)
umap.plot.show(p)
