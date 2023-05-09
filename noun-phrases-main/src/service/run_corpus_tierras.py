import os
import sys

ROOT_PATH = os.path.dirname(os.sep.join(os.path.abspath(__file__).split(os.sep)[:-2]))
sys.path.insert(1, ROOT_PATH)

from src.corpus.load_source import LoadSource
from src.corpus.build_source import BuildSource
from root import DIR_DATA

if __name__ == '__main__':
    path = DIR_DATA + "input/tierras/"
    ############ 1
    ls = LoadSource(db_name='sentences_tierras', db_coll='document', path=path, filter_files=[1])
    ls.load_files_tierras()
    ls.print_sentences_load()
    filter_pages_data = [2, 3] + [item for item in range(6, 9)] + [item for item in range(13, 19)] + \
                        [item for item in range(24, 34)] + [item for item in range(53, 58)]
    # filter_pages_data = []
    bs = BuildSource(load_files_object=ls, filter_pages=[{0: filter_pages_data}])
    # bs = BuildSource(load_files_object=ls)
    bs.parser_document()
    bs.print_sentences_parser()
    bs.analyzer_doc()
    # bs.graph_doc()
    # bs.compare_doc()
    # bs.network_doc()
    bs.save_db()

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


"""
Sentencia Los Naranjos - Drummond. Rad. 20001-31-21-002-2017-00027-00
Hechos: Pág. 2-3
La Oposición: Pág. 6-8 
Contexto de violencia: Pág. 13-18
Caso Concreto: Pág. 24-33
Resuelve: Pág. 53-57 (teniendo en cuenta las consideraciones expuestas en la reunión) 

Sentencia Urabá - Fondo Ganadero de Córdoba. Rad. 05045312100220150090901
Síntesis de fundamentos fácticos (numeral 2). Pág. 2-3
Tramite procesal y oposición (numeral 3). Pág. 3-4
El caso concreto: 5.1. De la relación jurídica con la tierra. 5.2. De la ruptura del vínculo material y jurídico y su relación o no con el conflicto armado. Pág 8-24
Restitución y formalización, órdenes de amparo e individualización del predio a restituir. Pág. 24-29
Falla. Pág. 29-34 (teniendo en cuenta las consideraciones expuestas en la reunión) 
"""