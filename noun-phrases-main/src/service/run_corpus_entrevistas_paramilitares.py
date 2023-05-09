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
outpath = DIR_DATA + "output/entrevistas_paramilitares"
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
types = [
    "verb",
    "person",
    "organisation",
    "adjective",
    "chunk",
    "noun-adj-pair",
    "place",
]

ids_entrevistas = {
    "magdalena_medio": [
        "195-VI-00021",
        "220-VI-00023",
        "195-VI-00004",
        "195-VI-00023",
        "195-VI-00027",
        "239-VI-00008",
        "240-AA-00001",
        "240_AA_00007",
        "252-AA-00006",
        "577-VI-00008",
        "220-VI-00031",
        "243-VI-00010",
        "193-AA-00001",
        "195-VI-00011",
        "444-VI-00012",
        "224-VI-OOO15",
        "195-VI-00006",
        "195-VI-00007",
        "195-VI-00009",
        "195-VI-00013",
        "220-VI-00048",
        "220-VI-00031",
        "239-VI-00028",
        "001-VI-00020",
    ],
    "casa_castaño": [
        "241-PR-00299",
        "241-AA-00001",
        "040-VI-00051",
        "077-VI-00010",
        "241-VI-00010",
        "427-VI-00010",
        "080-VI-00007",
        "426-VI-00012",
        "159-VI-00005",
        "077-PR-00443",
        "159-VI-00004",
        "426-VI-00020",
        "23-PR-00015",
        "123-PR-00478",
        "077-PR-00434",
        "084-PR-00402",
        "241-PR-00338",
        "041-VI-00007",
        "159-VI-00032",
        "427-VI-00008",
        "058-VI-00038",
        "185-PR-00423",
        "241-PR-00860",
        "311-PR-00633",
        "CIU-14711",
    ],
    "llanos_orientales": [
        "175-VI-00024",
        "163-VI-00002",
        "163-VI-00032",
        "163-VI-00020",
        "185-PR-00771",
        "185-PR-00009",
        "175-PR-00439",
        "098-PR-00354",
        "447-VI-00007",
        "175-VI-00034",
        "175-VI-00037",
        "737-VI-00015",
        "087-PR-02825",
        "084-PR-00004",
    ],
}

with open(os.path.join(path_all_entevistas, "load_source.p"), "rb") as f:
    ls = pickle.load(f)

bs = BuildSource(ls, spacy_model_name="es_core_news_lg")
bs.load_pickle(path_all_entevistas)

for caso, ids in ids_entrevistas.items():
    for type_ in types:
        bs.draw_frequent_words(
            type_,
            for_each_doc=False,
            documents_filter=ids,
            path=os.path.join(outpath, f"entrevistas_{caso}_freqs_{type_}.pdf"),
            topn=35,
        )

    for pair in pairs:
        graph_name = f"graphs_{caso}_{pair[0]}_vs_{pair[1]}.p"
        if graph_name not in outfiles:
            cooc = list(
                bs.generate_cooccurrence_graph(
                    *pair, for_each_doc=False, documents_filter=ids
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
            documents_filter=ids,
            path=os.path.join(
                outpath, f"entrevistas_{caso}_heatmap_{pair[0]}_vs_{pair[1]}.pdf"
            ),
            graphs=cooc,
            topn=35,
        )
        print("Heatmaps generated for", pair)
        bs.draw_graphs(
            *pair,
            for_each_doc=False,
            documents_filter=ids,
            path=os.path.join(
                outpath, f"entrevistas_{caso}_graph_{pair[0]}_vs_{pair[1]}.html"
            ),
            graphs=cooc,
            max_conns=50,
            plotly=False,
        )
        print("Graphs generated for", pair)
