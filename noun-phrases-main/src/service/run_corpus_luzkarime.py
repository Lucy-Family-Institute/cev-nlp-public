"""
This file depends on run_corpus_entrevistas.py
where all entrevistas are processed.
"""

import os
import sys
import pickle

ROOT_PATH = os.path.dirname(os.sep.join(os.path.abspath(__file__).split(os.sep)[:-2]))
sys.path.insert(1, ROOT_PATH)

from src.corpus.build_source_new import BuildSource
from root import DIR_DATA, DIR_OUTPUT

path = "/rep/repositories/dataset/survey/fichas_interviews.csv.gz"
path_all_entevistas = DIR_DATA + "output/entrevistas_all"
outpath = DIR_DATA + "output/luzkarime"
os.makedirs(outpath, exist_ok=True)
outfiles = os.listdir(outpath)

pairs = [
    ("person", "person"),
    ("organisation", "organisation"),
    ("person", "organisation"),
    ("person", "verb"),
    ("organisation", "verb"),
    ("organisation", "place"),
    ("chunk", "person"),
    ("chunk", "verb"),
    ("chunk", "organisation"),
    ("chunk", "adjective"),
    ("chunk", "place"),
]
types = [
    "verb",
    "person",
    "organisation",
    "adjective",
    "chunk",
    "noun-adj-pair",
    "place",
]

ids_entrevistas = [
    "056-VI-00016",
    "058-VI-00004",
    "070-VI-00002",
    "032-VI-00001",
    "109-VI-00004",
    "059-VI-00003",
    "059-VI-00010",
    "056-VI-00008",
    "056-VI-00009",
    "056-VI-00014",
    "059-VI-00004",
    "037-VI-00001",
    "109-VI-00002",
    "042-VI-00001",
    "040-VI-00001",
]

with open(os.path.join(path_all_entevistas, "load_source.p"), "rb") as f:
    ls = pickle.load(f)

bs = BuildSource(ls, spacy_model_name="es_core_news_lg")
bs.load_pickle(path_all_entevistas)

for type_ in types:
    bs.draw_frequent_words(
        type_,
        for_each_doc=False,
        documents_filter=ids_entrevistas,
        path=os.path.join(outpath, f"entrevistas_freqs_{type_}.pdf"),
        topn=35,
    )

for pair in pairs:
    graph_name = f"graphs_{pair[0]}_vs_{pair[1]}.p"
    if graph_name not in outfiles:
        cooc = list(
            bs.generate_cooccurrence_graph(
                *pair, for_each_doc=False, documents_filter=ids_entrevistas
            )
        )
        with open(os.path.join(outpath, graph_name), "wb") as f:
            pickle.dump(cooc, f)
    else:
        with open(os.path.join(outpath, graph_name), "rb") as f:
            cooc = pickle.load(f)
    bs.draw_heatmaps(
        *pair,
        for_each_doc=False,
        documents_filter=ids_entrevistas,
        path=os.path.join(
            outpath, f"entrevistas_heatmap_{pair[0]}_vs_{pair[1]}.pdf"
        ),
        graphs=cooc,
        topn=35,
    )
    print("Heatmaps generated for", pair)
    bs.draw_graphs(
        *pair,
        for_each_doc=False,
        documents_filter=ids_entrevistas,
        path=os.path.join(
            outpath, f"entrevistas_graph_{pair[0]}_vs_{pair[1]}.html"
        ),
        graphs=cooc,
        max_conns=50,
        plotly=False,
    )
    print("Graphs generated for", pair)
