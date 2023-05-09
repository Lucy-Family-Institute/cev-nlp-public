import os
import sys
import pandas as pd

ROOT_PATH = os.path.dirname(os.sep.join(os.path.abspath(__file__).split(os.sep)[:-2]))
sys.path.insert(1, ROOT_PATH)

from src.corpus.load_source import LoadSource
from src.corpus.build_source import BuildSource
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


if __name__ == '__main__':
    df = pd.read_excel(DIR_DATA + "input/tierras/Sentencias_Cesar.xlsx", usecols=[0, 1])
    df = df.dropna()
    path = DIR_DATA + "input/tierras/"
    names_sent = []
    pages_sent = []
    for idx, row in df.iterrows():
        ############ 1
        print(idx, row)
        filename = str(row["No Sentencia"]) + ".pdf"
        filter_pages_data = parse_pages(row["Páginas lectura"])
        names_sent.append(filename)
        pages_sent.append({idx: filter_pages_data})

    ls = LoadSource(db_name='sentences_tierras', db_coll='document', path=path, filter_file_names=names_sent)
    ls.load_files_tierras()
    # ls.print_sentences_load()
    # filter_pages_data = []
    bs = BuildSource(load_files_object=ls, filter_pages=pages_sent)
    # bs = BuildSource(load_files_object=ls)
    bs.parser_document()
    bs.print_sentences_parser()
    bs.analyzer_doc()
    try:
        bs.graph_docs()
    except Exception as e:
        print('graph_docs', e)
    try:
        bs.compare_docs(all_docs=True)
    except Exception as e:
        print('compare_docs', e)
    try:
        bs.network_docs(all_docs=True)
    except Exception as e:
        print('network_docs', e)
    try:
        bs.save_db()
    except Exception as e:
        print('save_db', e)
    try:
        bs.save()
    except Exception as e:
        print('save', e)
    print("Success! Hurrayyyyyyyy!!!")