import pandas as pd
import umap
import umap.plot

for min_labels in [2, 3, 4, 5, 6]:
    df = pd.read_excel(f"tortura_vec_representation_average_min_labels_{min_labels}.xlsx")
    vectors = df.iloc[:, 3:].to_numpy()

    reducer = umap.UMAP(n_neighbors=4, min_dist=0.1)
    embedding = reducer.fit_transform(vectors)

    umap.plot.output_file(f'tortura_map_min_labels_{min_labels}.html')
    p = umap.plot.interactive(reducer, hover_data=df[["interview_id", "text"]], point_size=5)
    umap.plot.show(p)
