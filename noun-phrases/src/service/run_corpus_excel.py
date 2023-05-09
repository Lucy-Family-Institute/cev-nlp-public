import os
import sys

ROOT_PATH = os.path.dirname(os.sep.join(os.path.abspath(__file__).split(os.sep)[:-2]))
sys.path.insert(1, ROOT_PATH)

from src.corpus.load_source import LoadSource
from src.corpus.build_source import BuildSource
from root import DIR_DATA, DIR_INPUT, DIR_OUTPUT

if __name__ == '__main__':
    ROOT_PATH = os.path.dirname(os.sep.join(os.path.abspath(__file__).split(os.sep)[:-3]))
    sys.path.insert(1, ROOT_PATH)
    path = DIR_INPUT + "other_excel/11052021 Dominio Transversal 4 RESISTENCIAS ORGANIZADAS.xlsx"
    ls = LoadSource(
        db_name="entrevistas_resistencias",
        db_coll="document",
        path=path,
        keyword="",
        filter_files=[],
    )
    ls.load_files_csv_from_interviews()
    ls.print_sentences_load()

    bs = BuildSource(load_files_object=ls)
    # bs = BuildSource(load_files_object=ls)
    bs.parser_document()
    bs.print_sentences_parser()
    bs.analyzer_doc()
    bs.save_db()
    bs.save()
    bs.graph_docs()
    bs.compare_docs(all_docs=True)
    bs.network_docs(all_docs=True)
    # bs.save_db()
    # bs.save()
    # bs = BuildSource.load(DIR_OUTPUT + '2021-04-18_21-48_model.pkl')
    # #bs.save_db()
    # #bs.compare_docs(all_docs=True)
    # bs.network_docs(all_docs=True)


    # ############ 2
    # ls = LoadSource(db_name='sentencias_jyp', db_coll='document', path=path, filter_files=[0])
    # ls.load_files_tierras()
    # ls.print_sentences_load()
    # filter_pages_data = [item for item in range(2, 4)] + [item for item in range(8, 34)]
    # bs = BuildSource(load_files_object=ls, filter_pages=[{0: filter_pages_data}])
    # bs.parser_document()
    # bs.print_sentences_parser()
    # bs.analyzer_doc()
    # # bs.graph_doc()
    # # bs.compare_doc()
    # # bs.network_doc()


