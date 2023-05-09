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
    if pages != pages:
        return []
    pages_list = []
    sections = pages.lower().replace("pag", "").replace(" ", "").replace(".", "").replace("(", "").replace(")", "").replace(",", "").split("/")
    for section in sections:
        if "-" in section:
            start, end = section.split("-")[-2:]
            if start == "" or end == "":
                print(f"Invalid section '{section}' found in pages '{pages}'")
                continue
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
    ("organisation", "place"),
]

types = ["verb", "person", "organisation", "adjective", "chunk", "noun-adj-pair", "place"]
df = pd.read_excel(DIR_DATA + "input/jyp/pagination.xlsx", usecols=["Nombre archivo", "Hechos relevantes ", "Antecedentes relevantes ", "Contextualización"])
df = df.dropna(subset=["Nombre archivo"])
path = DIR_DATA + "input/jyp/"
outpath = DIR_DATA + "output/sentencias_jyp"
os.makedirs(outpath, exist_ok=True)

all_filenames = []
page_filters = {}
for idx, row in df.iterrows():
    no_sent = str(row["Nombre archivo"])
    print(f"Processing sentence: {no_sent}")
    filename = no_sent + ".pdf"
    # FIXME: the following size restriction is only due to exploding RAM usage
    # Also, sometimes the name has strange characters and cannot be accessed from the filename in the excel file
    try:
        filesize = os.path.getsize(os.path.join(path, filename))
    except:
        continue
    if filesize < 50_000_000:
        all_filenames.append(filename)
        filter_pages_data = parse_pages(row["Hechos relevantes "])
        filter_pages_data += parse_pages(row["Antecedentes relevantes "])
        filter_pages_data += parse_pages(row["Contextualización"])
        page_filters[filename] = filter_pages_data

outfiles = os.listdir(outpath)
if "load_source.p" not in outfiles:
    ls = LoadSource(
        db_name="sentences_jyp",
        db_coll="document",
        path=path,
        filter_file_names=all_filenames,
    )
    ls.load_files_jyp()
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
        path=os.path.join(outpath, f"sentencias_jyp_freqs_{type_}.pdf"),
        topn=35,
    )
    # bs.draw_frequent_words(
    #     type_, for_each_doc=True, path=outpath, suffix=f"_freqs_{type_}", topn=20
    # )
    print("Frequent words generated for", type_)

for pair in pairs:
    graph_name = f"graphs_{pair[0]}_{pair[1]}.p"
    if graph_name not in outfiles:
        cooc = list(bs.generate_cooccurrence_graph(*pair, for_each_doc=False))
        cooc_for_each = list(bs.generate_cooccurrence_graph(*pair, for_each_doc=True))
        with open(os.path.join(outpath, graph_name), "wb") as f:
            pickle.dump([cooc, cooc_for_each], f)
    else:
        with open(os.path.join(outpath, graph_name), "rb") as f:
            cooc, cooc_for_each = pickle.load(f)

    bs.draw_heatmaps(
        *pair,
        for_each_doc=False,
        path=os.path.join(
            outpath, f"sentencias_jyp_heatmap_{pair[0]}_vs_{pair[1]}.pdf"
        ),
        graphs=cooc,
        topn=35,
    )
    # bs.draw_heatmaps(
    #     *pair,
    #     for_each_doc=True,
    #     path=outpath,
    #     suffix=f"heatmap_{pair[0]}_vs_{pair[1]}",
    #     graphs=cooc_for_each,
    #     topn=20,
    # )
    print("Heatmaps generated for", pair)
    bs.draw_graphs(
        *pair,
        for_each_doc=False,
        path=os.path.join(outpath, f"sentencias_jyp_graph_{pair[0]}_vs_{pair[1]}.html"),
        graphs=cooc,
        max_conns=50,
        plotly=False,
    )
    # bs.draw_graphs(
    #     *pair,
    #     for_each_doc=True,
    #     path=outpath,
    #     suffix=f"graph_{pair[0]}_vs_{pair[1]}",
    #     graphs=cooc_for_each,
    #     max_conns=50,
    # )
    print("Graphs generated for", pair)
    