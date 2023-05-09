import glob
import pdftotext
from datetime import datetime
import sys
from tqdm import tqdm
import os
import pandas as pd
import pathlib
from typing import List
import warnings
import re
import pytesseract
from pdf2image import convert_from_path

ROOT_PATH = os.path.dirname(os.sep.join(os.path.abspath(__file__).split(os.sep)[:-2]))
sys.path.insert(1, ROOT_PATH)

from src.util.data_acces import DataAccessMongo
from root import DIR_INPUT


class LoadSource(object):
    def __init__(
        self,
        path: pathlib.Path,
        db_name: str,
        db_coll: str,
        keyword: str = None,
        filter_files: list = [],
        filter_patterns: List[str] = [],
        filter_file_names: List[str] = [],
    ):
        """
        Creates a LoadSource object.

        Parameters
        ----------
        path: Path
            Path to a directory where files are stored.
        db_name: str
            Name of the Mongo database.
        db_coll: str
            Mongo collection of the database.
        keyword: str
            # TODO: ask Gabriel what is this
        filter_files: list
            List of integers picking the files to be taken into account.
            # ! I don't like this way of filtering files because the integers depend on
            # ! the order of files. If new files are included, the order might be changed.
        filter_patterns: List[str]
            A list of file patterns to match (not in regex)
            e.g. ['sentencia*-2020.pdf'] will match 'sentencia01-01-2020.pdf', and other
            files with the same pattern. You can pass many patterns to be matched.
        filter_file_names: List[str]
            Match specific file names.
        """
        print("A LoadSource object has been created")
        self.client_mongo = DataAccessMongo()
        self.db_name = db_name
        self.db_coll = db_coll
        self.path = os.path.abspath(path)
        # TODO: some of these should be passed to the methods to load jyp, tierras or entrevistas
        # thus they should not be class attributes
        self.keyword = keyword
        self.filter_files = filter_files
        self.filter_patterns = filter_patterns
        self.filter_file_names = filter_file_names
        self.list_files = []
        self.list_texts_corpus = []

        if len(filter_files) > 0:
            warnings.warn(
                "Filtering files with indices will be deprecated as some indices "
                "can be out of range, or they might not refer to the files you "
                "specifically want. Consider using `filter_patterns` and "
                "`filter_file_names` instead. See the documentation.",
                UserWarning,
            )

    def load_sentencias(
        self,
        extension_pattern: str = "*.pdf",
        to_replace: list = [],
        get_date: bool = False,
    ):
        all_files = glob.glob(os.path.join(self.path, extension_pattern))
        print(f"There are {len(all_files)} files detected")
        files_kept = []  # Relative paths saved here
        for pattern in self.filter_patterns:
            files_kept.extend(glob.glob(os.path.join(self.path, pattern)))
        for f in self.filter_file_names:
            f = os.path.join(self.path, f)
            if f not in all_files:
                print(f"File {f} not found in the path {self.path} with files {all_files}")
            else:
                files_kept.append(f)
        if len(self.filter_files) > 0:
            try:
                self.list_files = [all_files[item] for item in self.filter_files]
                if len(files_kept) > 0:
                    warnings.warn(
                        "Filtering by indices, patterns and filenames are all activated. "
                        "Falling back to filtering by indices only. Consider using only "
                        "filtering by patterns and filenames instead.",
                        UserWarning,
                    )
            except IndexError:
                warnings.warn(
                    "Indices provided when building the LoadSource object "
                    "are out of range, which raised an IndexError. Falling back "
                    "to using all detected files.",
                    UserWarning,
                )
        else:
            if len(files_kept) == 0:
                self.list_files = all_files
            else:
                self.list_files = files_kept
        print(f"There are {len(self.list_files)} files filtered.")

        # Load the files:
        for i in tqdm(range(len(self.list_files)), desc="Read Corpus"):
            single_file = self.list_files[i]
            # Process name:
            if get_date:
                name = single_file.split("_")
                date_time_str = name[1]
                for replace_pair in to_replace:
                    date_time_str = date_time_str.replace(*replace_pair)
                try:
                    date_time_obj = datetime.strptime(date_time_str, "%d-%b-%y")
                except ValueError:
                    print(
                        "Error produced in the following file, where date could not be formatted"
                    )
                    print(single_file)
                    continue
                date_dict = {
                    "date": date_time_obj,
                    "year": date_time_obj.year,
                    "month": date_time_obj.month,
                    "day": date_time_obj.day,
                }
            else:
                date_dict = {"date": None, "year": None, "month": None, "day": None}

            # Get text:
            with open(single_file, "rb") as f:
                pdf = pdftotext.PDF(f)

            text_raw = "\n\n".join(pdf)
            result = re.sub(r'[\W_]+', '', text_raw)
            if len(result) == 0:
                pdf = self.ocr_extract_text(single_file)

            single_dict = {
                "path": single_file,
                "file_name": single_file.split("/")[-1].split(".")[0],
                "num_pages": len(pdf),
                "pages_raw": list(pdf),
                "text_raw": "\n\n".join(pdf),
                "sentences": [],
                "parsed_doc": [],
                "identifier": single_file,
            }
            single_dict.update(date_dict)
            self.list_texts_corpus.append(single_dict)

    @staticmethod
    def ocr_extract_text(file_path):
        list_page = []
        pages = convert_from_path(file_path, 500)
        for pageNum, imgBlob in enumerate(pages):
            text = pytesseract.image_to_string(imgBlob, lang='eng+spa+fra')
            list_page.append(text)
        return list_page

    def load_files_jyp(self):
        to_replace = [
            (".pdf", ""),
            ("dic", "dec"),
            ("agt", "aug"),
            ("abr", "apr"),
            ("ago", "aug"),
            ("ene", "jan"),
            ("2014", "14"),
        ]
        self.load_sentencias(to_replace=to_replace, get_date=True)

    def load_files_tierras(self):
        self.load_sentencias()

    def load_files_entrevistas(self):
        # The following imports are needed to correctly read the huge interviews
        import sys
        import csv

        csv.field_size_limit(sys.maxsize)

        # TODO: self.keyword is a very simple filter with fichas. Consider expanding this one to a list of dictinaries of the form {keyword: match_value}
        df_interviews = pd.read_csv(
            self.path,
            sep="¬",
            compression="gzip",
            usecols=["interview_id", "fecha_inicio", "text", self.keyword],
        )
        # 'v_m_masacre', 'interview_id', 'text', 'id_victima',
        df_text = df_interviews[df_interviews[self.keyword] == 1].drop_duplicates()
        # resetting index
        df_text.reset_index(inplace=True)
        print("Shape of loaded interviews dataframe:", df_text.shape)

        # Load
        count = 0
        if len(self.filter_files) > 0:
            try:
                df_text = df_text.iloc[self.filter_files]
            except IndexError:
                warnings.warn(
                    "Indices provided when building the LoadSource object "
                    "are out of range, which raised an IndexError. Falling back "
                    "to using all detected files.",
                    UserWarning,
                )

        def parse_date(date, start, end):
            return date[start:end]

        df_text["year"] = df_text.fecha_inicio.apply(lambda x: parse_date(x, 0, 4))
        df_text["month"] = df_text.fecha_inicio.apply(lambda x: parse_date(x, 5, 7))
        df_text["day"] = df_text.fecha_inicio.apply(lambda x: parse_date(x, 8, 10))
        df_text["path"] = self.path
        df_text["num_pages"] = 1
        df_text["pages_raw"] = df_text.text.apply(lambda x: [x])
        df_text["sentences"] = df_text.text.apply(lambda x: [])
        df_text["parsed_doc"] = df_text.text.apply(lambda x: [])
        df_text["file_name"] = df_text.interview_id
        df_text["identifier"] = df_text.interview_id

        df_text = df_text[
            [
                "path",
                "file_name"
                "fecha_inicio",
                "year",
                "month",
                "day",
                "num_pages",
                "pages_raw",
                "text",
                "sentences",
                "parsed_doc",
                "interview_id",
                "identifier",
            ]
        ]
        df_text = df_text.rename(columns={"fecha_inicio": "date", "text": "text_raw"})

        self.list_texts_corpus.extend(df_text.to_dict(orient="records"))

    def load_files_csv_from_interviews(self):
        # The following imports are needed to correctly read the huge interviews
        import sys
        import csv

        csv.field_size_limit(sys.maxsize)

        # TODO: self.keyword is a very simple filter with fichas. Consider expanding this one to a list of dictinaries of the form {keyword: match_value}
        df_text = pd.read_excel(
            self.path,
            usecols=["codigo_entrevista", "entrevista_fecha", "texto_marcado"]
        )
        # 'v_m_masacre', 'interview_id', 'text', 'id_victima',
        # df_text = df_interviews[df_interviews[self.keyword] == 1].drop_duplicates()
        # # resetting index
        # df_text.reset_index(inplace=True)
        print("Shape of loaded interviews dataframe:", df_text.shape)

        # Load
        count = 0
        if len(self.filter_files) > 0:
            try:
                df_text = df_text.iloc[self.filter_files]
            except IndexError:
                warnings.warn(
                    "Indices provided when building the LoadSource object "
                    "are out of range, which raised an IndexError. Falling back "
                    "to using all detected files.",
                    UserWarning,
                )

        def parse_date(date, start, end):
            # print(date, date[start:end])
            return date[start:end]

        df_text["year"] = df_text.entrevista_fecha.apply(lambda x: parse_date(x, 6, 10))
        df_text["month"] = df_text.entrevista_fecha.apply(lambda x: parse_date(x, 3, 5))
        df_text["day"] = df_text.entrevista_fecha.apply(lambda x: parse_date(x, 0, 2))
        df_text["date"] = df_text.entrevista_fecha
        df_text["path"] = self.path
        df_text["num_pages"] = 1
        df_text["pages_raw"] = df_text.texto_marcado.apply(lambda x: [x])
        df_text["sentences"] = df_text.texto_marcado.apply(lambda x: [])
        df_text["doc_parser"] = df_text.texto_marcado.apply(lambda x: [])
        df_text["file_name"] = df_text.codigo_entrevista
        df_text["text"] = df_text.texto_marcado

        df_text = df_text[
            [
                "path",
                "file_name",
                "date",
                "entrevista_fecha",
                "year",
                "month",
                "day",
                "num_pages",
                "pages_raw",
                "text",
                "sentences",
                "doc_parser",
                "codigo_entrevista",
            ]
        ]
        df_text = df_text.rename(columns={"fecha_inicio": "date", "text": "text_raw"})

        self.list_texts_corpus.extend(df_text.to_dict(orient="records"))

    def print_sentences_load(self):
        count = 0
        for item in self.list_texts_corpus:
            count += 1
            # print('-> ', count, '\tfile_name:', item['file_name'][:50].zfill(50).replace('0', ' '), '\tdate:',
            #       item['date'], '\tnum_pages:', item['num_pages'], len(item['pages_raw']))
            print(
                item["path"],
                item["date"],
                item["year"],
                item["month"],
                item["day"],
                item["num_pages"],
                item["text_raw"][:20],
            )
            if count % 10 == 0:
                break


if __name__ == "__main__":
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

    # path = "/home/vvargas/repositories/noun-phrases/data/input/jyp/"
    # ls = LoadSource(
    #     db_name="sentencias_jyp", db_coll="document", path=path, filter_files=[0, 1]
    # )
    # ls.load_files_jyp()
    # ls.print_sentences_load()
    #
    # print("Testing tierras...")
    # path = "/home/vvargas/repositories/noun-phrases/data/input/tierras/"
    # ls = LoadSource(
    #     db_name="sentencias_tierras", db_coll="document", path=path, filter_files=[0]
    # )
    # ls.load_files_tierras()
    # ls.print_sentences_load()
