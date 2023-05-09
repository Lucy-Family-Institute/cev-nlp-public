import os
import sys
import pandas as pd
import pickle

ROOT_PATH = os.path.dirname(os.sep.join(os.path.abspath(__file__).split(os.sep)[:-2]))
sys.path.insert(1, ROOT_PATH)

from src.corpus.load_source_new import LoadSource
from src.corpus.build_source_new import BuildSource
from root import DIR_DATA


def parse_pages(pages):
    pages_list = []
    sections = pages.replace(" ", "").replace(".", "").split(";")
    for section in sections:
        if "-" in section:
            start, end = section.split("-")
            pages_list.extend(list(range(int(start), int(end) + 1)))
        elif section != "":
            pages_list.append(int(section))
    return pages_list


pairs = [
    ("person", "person"),
    ("organisation", "organisation"),
    ("person", "organisation"),
    ("person", "verb"),
    ("organisation", "verb"),
]
types = ["verb", "person", "organisation", "adjective", "chunk", "noun-adj-pair"]
df = pd.read_excel(DIR_DATA + "input/tierras/Sentencias_Cesar.xlsx", usecols=[0, 1])
df = df.dropna()
path = DIR_DATA + "input/tierras/"
outpath = DIR_DATA + "output/sentencias_cesar"
os.makedirs(outpath, exist_ok=True)

all_filenames = []
page_filters = {}
for idx, row in df.iterrows():
    no_sent = str(row["No Sentencia"])
    print(f"Processing sentence: {no_sent}")
    filename = no_sent + ".pdf"
    all_filenames.append(filename)
    filter_pages_data = parse_pages(row["Páginas lectura"])
    page_filters[filename] = filter_pages_data

outfiles = os.listdir(outpath)
if "load_source.p" not in outfiles:
    ls = LoadSource(
        db_name="sentences_tierras",
        db_coll="document",
        path=path,
        filter_file_names=all_filenames,
    )
    ls.load_files_tierras()
    with open(os.path.join(outpath, "load_source.p"), "wb") as f:
        pickle.dump(ls, f)
else:
    with open(os.path.join(outpath, "load_source.p"), "rb") as f:
        ls = pickle.load(f)

bs = BuildSource(ls, spacy_model_name="es_core_news_lg")
if "build_source.p" not in outfiles:
    bs.parse_documents(page_filters)
    bs.build_big_table()
    bs.build_big_graph()
    bs.save_pickle(outpath)
else:
    bs.load_pickle(outpath)

for type_ in types:
    bs.draw_frequent_words(
        type_,
        for_each_doc=False,
        path=os.path.join(outpath, f"all_sentencias_freqs_{type_}.pdf"),
        topn=20,
    )
#     bs.draw_frequent_words(
#         type_, for_each_doc=True, path=outpath, suffix=f"_freqs_{type_}", topn=20
#     )
#     print("Frequent words generated for", type_)
# for pair in pairs:
#     if "graphs.p" not in outfiles:
#         cooc = list(bs.generate_cooccurrence_graph(*pair, for_each_doc=False))
#         cooc_for_each = list(bs.generate_cooccurrence_graph(*pair, for_each_doc=True))
#         with open(os.path.join(outpath, "graphs.p"), "wb") as f:
#             pickle.dump([cooc, cooc_for_each], f)
#     else:
#         with open(os.path.join(outpath, "graphs.p"), "rb") as f:
#             cooc, cooc_for_each = pickle.load(f)

#     bs.draw_heatmaps(
#         *pair,
#         for_each_doc=False,
#         path=os.path.join(
#             outpath, f"all_sentencias_heatmap_{pair[0]}_vs_{pair[1]}.pdf"
#         ),
#         graphs=cooc,
#         topn=20,
#     )
#     bs.draw_heatmaps(
#         *pair,
#         for_each_doc=True,
#         path=outpath,
#         suffix=f"heatmap_{pair[0]}_vs_{pair[1]}",
#         graphs=cooc_for_each,
#         topn=20,
#     )
#     print("Heatmaps generated for", pair)
#     bs.draw_graphs(
#         *pair,
#         for_each_doc=False,
#         path=os.path.join(outpath, f"all_sentencias_graph_{pair[0]}_vs_{pair[1]}.html"),
#         graphs=cooc,
#         max_conns=50,
#     )
#     bs.draw_graphs(
#         *pair,
#         for_each_doc=True,
#         path=outpath,
#         suffix=f"graph_{pair[0]}_vs_{pair[1]}",
#         graphs=cooc_for_each,
#         max_conns=50,
#     )
#     print("Graphs generated for", pair)
