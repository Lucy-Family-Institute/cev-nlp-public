import os
import sys
import pickle

ROOT_PATH = os.path.dirname(os.sep.join(os.path.abspath(__file__).split(os.sep)[:-2]))
sys.path.insert(1, ROOT_PATH)

from src.corpus.load_source_new import LoadSource
from src.corpus.build_source_new import BuildSource
from root import DIR_DATA, DIR_OUTPUT

path = "/rep/repositories/dataset/survey/fichas_interviews.csv.gz"
outpath = DIR_DATA + "output/entrevistas_all"
os.makedirs(outpath, exist_ok=True)
outfiles = os.listdir(outpath)

pairs = [
    ("person", "person"),
    ("organisation", "organisation"),
    ("person", "organisation"),
    ("person", "verb"),
    ("organisation", "verb"),
    ("organisation", "place"),
]
types = ["verb", "person", "organisation", "adjective", "chunk", "noun-adj-pair", "place"]

if "load_source.p" not in outfiles:
    ls = LoadSource(db_name='entrevistas_all', db_coll='document', path=path)
    ls.load_files_entrevistas()
    ls.print_sentences_load()
    with open(os.path.join(outpath, "load_source.p"), "wb") as f:
        pickle.dump(ls, f)
else:
    with open(os.path.join(outpath, "load_source.p"), "rb") as f:
        ls = pickle.load(f)

bs = BuildSource(ls, spacy_model_name="es_core_news_lg")
if "build_source.p" not in outfiles:
    bs.parse_documents()
    bs.build_big_table()
    bs.build_big_graph()
    bs.save_pickle(outpath)
else:
    bs.load_pickle(outpath)

for type_ in types:
    bs.draw_frequent_words(
        type_,
        for_each_doc=False,
        path=os.path.join(outpath, f"all_entrevistas_freqs_{type_}.pdf"),
        topn=35,
    )

for pair in pairs:
    graph_name = f"graphs_{pair[0]}_{pair[1]}.p"
    if graph_name not in outfiles:
        cooc = list(bs.generate_cooccurrence_graph(*pair, for_each_doc=False))
        with open(os.path.join(outpath, graph_name), "wb") as f:
            pickle.dump(cooc, f)
    else:
        with open(os.path.join(outpath, graph_name), "rb") as f:
            cooc = pickle.load(f)
    bs.draw_heatmaps(
        *pair,
        for_each_doc=False,
        path=os.path.join(
            outpath, f"all_entrevistas_heatmap_{pair[0]}_vs_{pair[1]}.pdf"
        ),
        graphs=cooc,
        topn=35,
    )
    print("Heatmaps generated for", pair)
    bs.draw_graphs(
        *pair,
        for_each_doc=False,
        path=os.path.join(outpath, f"all_entrevistas_graph_{pair[0]}_vs_{pair[1]}.html"),
        graphs=cooc,
        max_conns=50,
        plotly=False,
    )
    print("Graphs generated for", pair)