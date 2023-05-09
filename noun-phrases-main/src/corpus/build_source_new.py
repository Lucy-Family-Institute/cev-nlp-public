from typing import List, Dict, Tuple, Any
from itertools import combinations
import re
import pickle
import os
import warnings
import numpy as np
import sys
import spacy
from spacy.matcher import Matcher
import pandas as pd
from typeguard import typechecked
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import networkx as nx
from pyvis.network import Network
from interviews.graph.graph import nx2pyvis
from preprocess import (
    tokenise_sentences,
    proper_encoding,
    remove_punctuation,
    PreprocessPipeline,
)
from preprocess.cev import identify_tags, parse_tag, delete_tags

ROOT_PATH = os.path.dirname(os.sep.join(os.path.abspath(__file__).split(os.sep)[:-2]))
sys.path.insert(1, ROOT_PATH)

from src.corpus.plotly_graphs_utils import frequency_graph, heatMap, net


class BuildSource(object):
    # TODO these are not used for anything...

    verb_patterns = [
        {"POS": "VERB", "OP": "*"},
        {"POS": "ADV", "OP": "*"},
        {"POS": "VERB", "OP": "+"},
        {"POS": "PART", "OP": "*"},
    ]

    entity_type_dict = {"PER": "person", "ORG": "organisation", "LOC": "place"}

    def __init__(self, source, spacy_model_name="es_core_news_md"):
        """
        Constructs a BuildSource object, which processes the texts in
        a previously created LoadSource object.

        Parameters
        ----------
        source : LoadSource
            A LoadSource object
        spacy_model_name : str
            Name of a spaCy model (either a default model) or some path
            to a modified spaCy model.
        """
        self.source = source
        self.nlp = spacy.load(spacy_model_name)
        self.preprocess = PreprocessPipeline(lemmatise=False, tokenise=False)
        self.matcher = Matcher(self.nlp.vocab)
        self.matcher.add("NounAdjPairs", [[{"POS": "NOUN"}, {"POS": "ADJ"}]])
        self.matcher.add("Adjectives", [[{"POS": "ADJ"}]])
        self.matcher.add("Verbs", [[{"POS": "VERB"}]])
        # Holder for a pandas dataframe with text features
        self.big_table = None
        # Holder for a list of identifiers of parsed documents
        self.parsed_documents_ids = []
        # Holder for a networkx graph, which is a graphical representation of big_table
        self.big_graph = None

    @typechecked
    def save_pickle(self, path: str):
        """
        Saves the data-related attributes of the `BuildSource`
        object to a pickle file.

        Parameters
        ----------
        path : str
            A path to a directory where a datafile will be saved
        """
        with open(os.path.join(path, "build_source.p"), "wb") as f:
            pickle.dump([self.big_table, self.parsed_documents_ids, self.big_graph], f)

    @typechecked
    def load_pickle(self, path: str):
        """
        Loads the data-related attributes of the `BuildSource`
        object from a pickle file.

        Parameters
        ----------
        path : str
            A path to a directory where a datafile will be loaded
        """
        with open(os.path.join(path, "build_source.p"), "rb") as f:
            self.big_table, self.parsed_documents_ids, self.big_graph = pickle.load(f)
        print("Big table shape is", self.big_table.shape)
        print("Number of parsed documents is", len(self.parsed_documents_ids))
        print(
            "Number of nodes and edges of the graph are",
            self.big_graph.number_of_nodes(),
            self.big_graph.number_of_edges(),
        )

    @typechecked
    def parse_documents(
        self,
        filter_pages: Dict[str, List[int]] = None,
        stopwords: Dict[str, List[str]] = {},
    ):
        """
        Parses all documents in `self.source.list_texts_corpus`.

        Parameters
        ----------
        filter_pages : Dict[str, List[int]], optional
            Consider the fact that paths are unique for each document
            in `self.source.list_texts_corpus`. For this reason,
            the `filter_pages` argument admits a dictionary whose key
            is any substring that uniquely identifies a document from
            its path, and whose value is a list of 1-indexed pages to
            parse from each document.

            An example of how this works is: suppose that in the source
            corpus there are 10 documents with paths of the form
            ".../sentencia_index.pdf", where index is a number that runs
            from A01 to A10. Thus an element of `filter_pages` could
            be `"A02": [1, 2]`, or equivalently `"sentencia_A02": [1, 2]`.
            Both of those elements will match the document with the path
            ".../sentencia_A02.pdf", and only pages 1 and 2 will be used.

        stopwords : Dict[str, List[str]]
            A dictionary where keys are entity types (e.g. "PER", "ORG",
            "LOC") and values are lists of words never to be included
            as these entities. This essentially is a hard-coding way to
            avoid wrong detected entities.
        """
        self.parsed_documents_ids = []
        for single_doc in tqdm(self.source.list_texts_corpus, desc="Structure Corpus"):
            document_in_filter = False
            if filter_pages is not None:
                for path_segment, pages_in_filter in filter_pages.items():
                    if path_segment in single_doc["path"]:
                        document_in_filter = True
                        break
                filter_pages.pop(path_segment)
            else:
                document_in_filter = True
                pages_in_filter = []
            if not document_in_filter:
                continue
            else:
                self.parsed_documents_ids.append(single_doc["identifier"])
            page_counter = 0
            sentence_counter = 0
            single_doc["doc_parser"] = []
            if len(pages_in_filter) > 0:
                try:
                    pages_raw = [
                        (page_num - 1, single_doc["pages_raw"][page_num - 1])
                        for page_num in pages_in_filter
                    ]
                except IndexError:
                    print(
                        f"You have a filter with the pages {pages_in_filter} "
                        f"But the document has {len(single_doc['pages_raw'])} pages."
                    )
                    continue
            else:
                pages_raw = list(enumerate(single_doc["pages_raw"]))

            for page_num, page_raw in pages_raw:
                page_counter += 1
                local_sentence_counter = 0
                page_text = re.sub("\n+", " ", page_raw)
                page_text = re.sub("\s+", " ", page_text)
                sentences_raw = tokenise_sentences(page_text)
                parsed_sentences = []
                for sentence_raw in sentences_raw:
                    sentence_raw = proper_encoding(sentence_raw)
                    sentence_dict = {
                        "page_num": page_num,
                        "global_sentence_num": sentence_counter,
                        "local_sentence_counter": local_sentence_counter,
                        "raw_text": sentence_raw,
                        "chunks": [],
                        "verbs": [],
                        "entities": [],
                        "adjectives": [],
                        "noun_adjective_pairs": [],
                        "tags": [],
                    }
                    local_sentence_counter += 1
                    parsed_tags = self._get_tags(sentence_raw)
                    # TODO: check whether we are dealing with interviews.
                    # If so: pass is_interview=True to remove_text_dirt.
                    # Else, pass False.
                    semi_raw_text = self._remove_text_dirt(sentence_raw)
                    sentence_dict["tags"].extend(parsed_tags)
                    sentence_dict["clean_text"] = semi_raw_text
                    doc = self.nlp(semi_raw_text)
                    chunk_counter = 0
                    for chunk in doc.noun_chunks:
                        processed_chunk = self._clean_text(chunk)
                        if processed_chunk != "":
                            sentence_dict["chunks"].append(
                                {
                                    # TODO is chunk_num useful for something at all?
                                    "chunk_num": chunk_counter,
                                    "type": "chunk",
                                    "info": {"text": processed_chunk},
                                    "original": chunk.text,
                                }
                            )
                            chunk_counter += 1
                    self._process_entities(doc, sentence_dict, stopwords)
                    adjectives, verbs, noun_adj_pairs = self._get_noun_adj_pairs(doc)
                    sentence_dict["adjectives"].extend(adjectives)
                    sentence_dict["verbs"].extend(verbs)
                    sentence_dict["noun_adjective_pairs"].extend(noun_adj_pairs)
                    single_doc["sentences"].append(sentence_dict)
                    parsed_sentences.append(sentence_dict)
                single_doc["parsed_doc"].append(
                    {"page_num": page_num, "sentences": parsed_sentences}
                )

    def _process_entities(self, doc, sentence_dict, stopwords={}):
        """
        Processes entities identified by spaCy model and saves
        them in a dictionary in-place.

        Parameters
        ----------
        doc
            A spaCy document
        sentence_dict : dict
            A dictionary with a key "entities" where dictionaries
            will be saved.
        stopwords : dict
            A dictionary where keys are entity types (e.g. "PER", "ORG", "LOC")
            and values are lists of words never to be included as these entities.
            This essentially is a hard-coding way to avoid wrong detected entities.
        """
        for entity in doc.ents:
            _type = entity.label_
            processed_entity = self._clean_text(entity)
            if processed_entity == "":
                continue
            flag_continue = False
            for ent_type, words in stopwords.items():
                if _type == ent_type:
                    if processed_entity in words:
                        flag_continue = True
                        break
            if flag_continue:
                continue
            sentence_dict["entities"].append(
                {
                    "type": _type,
                    "info": {"text": processed_entity},
                    "original": entity.text,
                }
            )

    @staticmethod
    def _get_tags(text: str) -> List[dict]:
        """
        Private method to get interview tags from text.

        Parameters
        ----------
        text : str
            Any text from an interview.

        Returns
        -------
        List[dict]
            A list of dictionary. One dictionary for each tag found within
            the `text`. The general fields of the dictionary are "type",
            indicating the type of the tag, "original", indicating the original
            text within the tag, and "info", which might be a string (containing
            an error) or a dictionary that contains further information about
            the tag. Different information is found depending on the type of the
            tag. See `preprocess.cev.parse_tag` for more information.
        """
        tags = identify_tags(text)
        return [parse_tag(t, verbose=False) for t in tags]

    @staticmethod
    def _remove_text_dirt(text: str, is_interview=True) -> str:
        """
        Private method to clean any given sentence from tags,
        interviewer-interviewee headers and punctuation.

        Parameters
        ----------
        text : str
            Any text.
        is_interview : bool
            Whether the text comes from an interview. There are a couple of
            preprocessing steps carried out on interviews that are important.
            Setting this to `False` will not carry out these steps (tag
            removal and interviewer-interviewee headers removal).

        Returns
        -------
        str
            Cleaned text.
        """
        if is_interview:
            text = delete_tags(text)
            text = (
                text.replace("ENT:", "")
                .replace("ENT2:", "")
                .replace("TEST:", "")
                .replace("TEST2:", "")
                .replace("[]", "")
            )
        return remove_punctuation(text)

    def _clean_text(self, doc_span) -> str:
        """
        Private method to apply preprocessing on a spaCy document
        or a span of a spaCy document.

        Parameters
        ----------
        doc_span
            A spaCy document or span of a document.

        Returns
        -------
        str
            Cleaned text.
        """
        lemmas = " ".join([w.lemma_ for w in doc_span])
        return self.preprocess(lemmas).strip()

    def _get_noun_adj_pairs(self, doc):
        """
        Private method to extract noun-adjective pairs from a spaCy document.

        Parameters
        ----------
        doc
            A spaCy document.

        Returns
        -------
        Tuple[List[Dict]]
            A tuple containing information about adjectives, verbs and
            noun-adjective pairs. Each of them is a list containing
            dictionaries: one dictionary for each occurrence. The fields
            of each dictionary are "text" which contain the clean text
            of the identified adj/verb/noun-adj pair, and "original",
            which contains the raw identified adj/verb/noun-adj pair.
        """
        matches = self.matcher(doc, as_spans=True)
        all_adjectives = []
        all_verbs = []
        noun_adj_pairs = []
        for span in matches:
            clean_span = self._clean_text(span)
            if clean_span == "":
                continue
            label = span.label_
            span_dict = {"text": clean_span, "original": span.text}
            if label == "NounAdjPairs":
                noun_adj_pairs.append(span_dict)
            elif label == "Adjectives":
                all_adjectives.append(span_dict)
            elif label == "Verbs":
                all_verbs.append(span_dict)
        return all_adjectives, all_verbs, noun_adj_pairs

    def print_parsed_sentences(self):
        for i, single_doc in enumerate(self.source.list_texts_corpus):
            print(
                f"Document number: {i}",
                f"Document filename: {single_doc['file_name']}",
                single_doc["parsed_doc"][0],
                sep="\n",
            )

    @typechecked
    def build_big_table(self, external_table: pd.DataFrame = None):
        """
        Builds a big table where identified features from the text are
        laid out. The columns of the big pandas table are:

        - type: can be verb, adjective, person, organisation, etc.
        - text: the text of the feature itself.
        - page_num: number of page where the feature appears in.
        - rel_sent_num: number of the relative sentence where the feature
            appears in.
        - doc_id: a unique identifier of the document (can be a filepath for
            sentencias, or a code for interviews)

        This method should be run after `parse_documents` has been run.

        Parameters
        ----------
        external_table : pandas.DataFrame
            A pandas dataframe. If this is provided, then the method will
            expand the information of external_table with information from
            the current documents being processed.
        """
        records = {
            "type": [],
            "text": [],
            "page_num": [],
            "rel_sent_num": [],
            "doc_id": [],
        }
        if external_table is not None:
            assert set(external_table.columns) == set(records.keys()), (
                "external_table must contain the following columns:\n"
                "- type\n- text\n- page_num\n- rel_sent_num\n- doc_id\n"
                f"The provided columns are: {external_table.columns}."
            )
        for single_doc in self.source.list_texts_corpus:
            identifier = single_doc["identifier"]
            for sentence in single_doc["sentences"]:
                fill_table = lambda _type, _text: self._fill_big_table_data(
                    records,
                    _type,
                    identifier,
                    sentence["page_num"],
                    sentence["local_sentence_counter"],
                    _text,
                )
                chunks = list(
                    filter(
                        lambda t: True
                        if len(t["info"]["text"].split()) in [2, 3, 4]
                        else False,
                        sentence["chunks"],
                    )
                )
                for chunk in chunks:
                    fill_table(chunk["type"], chunk["info"]["text"])
                for tag in sentence["tags"]:
                    if tag["type"] == "INC":
                        fill_table("tag_INC", tag["info"]["text"])
                for entity in sentence["entities"]:
                    ent_type = entity["type"]
                    if ent_type in self.entity_type_dict.keys():
                        fill_table(
                            self.entity_type_dict[ent_type], entity["info"]["text"]
                        )
                for verb in sentence["verbs"]:
                    fill_table("verb", verb["text"])
                for adj in sentence["adjectives"]:
                    fill_table("adjective", adj["text"])
                for pair in sentence["noun_adjective_pairs"]:
                    fill_table("noun-adj-pair", pair["text"])

        df = pd.DataFrame(
            {
                "type": pd.Series(records["type"]).astype("category"),
                "text": pd.Series(records["text"]),
                "page_num": pd.Series(records["page_num"]),
                "rel_sent_num": pd.Series(records["rel_sent_num"]),
                "doc_id": pd.Series(records["doc_id"]).astype("category"),
            }
        )
        if external_table:
            df = external_table.append(df)
        self.big_table = df

    @staticmethod
    def _fill_big_table_data(records, _type, _id, _page_num, _rel_sent_num, _text):
        records["type"].append(_type)
        records["doc_id"].append(_id)
        records["page_num"].append(_page_num)
        records["rel_sent_num"].append(_rel_sent_num)
        records["text"].append(_text)

    @typechecked
    def build_big_graph(self, external_graph: nx.Graph = None):
        """
        Builds a graph where nodes are text features with properties
        associatedto type of the feature, document, page number and
        relative sentence number where the feature appears, and its
        text. The node name is the text whereas the rest information
        is saved as an element of a list called "appears_in". Edges
        exist if nodes co-occur somewhere in the corpus. Edges have
        a property "appears_in" saving where the nodes co-occur.

        This is run from `self.big_table`, so `build_big_table` needs
        to be run.

        Parameters
        ----------
        external_graph : networkx.Graph
            A networkX graph created in a previous instance of `BuildSource`.
        """
        if self.big_table is None:
            warnings.warn(
                "Big Table has not been built yet. "
                "Since this is necessary for building the big graph, "
                "I will build it for you. Consider running `build_big_table` "
                "before running `build_big_graph`.",
                UserWarning,
            )
            self.build_big_table()
        if external_graph is None:
            graph = nx.Graph()
        else:
            graph = external_graph
        for (doc, page, sent), group in self.big_table.groupby(
            by=["doc_id", "page_num", "rel_sent_num"]
        ):
            group = group[["text", "type"]].drop_duplicates()
            common_dict = {"doc_id": doc, "page_num": page, "rel_sent_num": sent}
            visited_nodes = []
            for node1, node2 in combinations(group.values, 2):
                node1_type = node1[1]
                node2_type = node2[1]
                node1_text = "_".join(node1)
                node2_text = "_".join(node2)

                if node1_text in graph.nodes:
                    if node1_text not in visited_nodes:
                        graph.nodes[node1_text]["appears_in"].append(common_dict)
                        visited_nodes.append(node1_text)
                else:
                    graph.add_node(
                        node1_text, appears_in=[common_dict], type=node1_type
                    )
                    visited_nodes.append(node1_text)
                if node2_text in graph.nodes:
                    if node2_text not in visited_nodes:
                        graph.nodes[node2_text]["appears_in"].append(common_dict)
                        visited_nodes.append(node2_text)
                else:
                    graph.add_node(
                        node2_text, appears_in=[common_dict], type=node2_type
                    )
                    visited_nodes.append(node2_text)
                if (node1_text, node2_text) not in graph.edges:
                    graph.add_edge(node1_text, node2_text, appears_in=[common_dict])
                else:
                    graph.edges[(node1_text, node2_text)]["appears_in"].append(
                        common_dict
                    )
        self.big_graph = graph

    @typechecked
    def generate_cooccurrence_graph(
        self,
        type1: str,
        type2: str,
        for_each_doc: bool = True,
        documents_filter: List[str] = None,
        stopwords: Dict[str, List[str]] = {},
    ):
        """
        Builds a co-occurrence graph where edges have a weight corresponding
        to the co-ocurrence, and the nodes are text features. Only nodes
        with `type1` will be connected to nodes of `type2`. To run, it is
        necessary that `self.big_graph` was built throguh `build_big_graph`.

        Parameters
        ----------
        type1 : str
            Features have types (see `BuildSource.build_big_table`'s
            documentation). Nodes of this type will be paired to nodes
            of type `type2`.
        type2 : str
            See `type1`.
        for_each_doc : bool
            A flag that indicates whether the method will return just
            one large co-ocurrence matrix corresponding to all the
            documents found in `self.parsed_documents_ids` (with some
            possible filters, see `documents_filter` argument), or
            a co-ocurrence matrix will be build for each document
            found in `self.parsed_documents_ids`.
        documents_filter : List[str]
            A list of document ids to filter documents.
        stopwords : Dict[str, List[str]]
            See documentation of BuildSource.parse_documents

        Yields
        -------
        networkx.Graph
            A weighted graph containing information about co-occurrences.
        str
            A document id
        """
        if documents_filter is not None:
            doc_ids = []
            for doc_filter in documents_filter:
                for parsed_doc in self.parsed_documents_ids:
                    if doc_filter in parsed_doc:
                        doc_ids.append(parsed_doc)
                        break
            filter_is_none = False
        else:
            doc_ids = self.parsed_documents_ids
            filter_is_none = True

        print(
            "Generating coocurrence graphs... there are",
            len(doc_ids),
            "to be generated",
        )
        print("An example of a doc id is", doc_ids[0])

        if for_each_doc:
            for doc_id in doc_ids:
                yield self._build_cooc_graph(
                    type1, type2, doc_id, stopwords=stopwords
                ), doc_id
        else:
            cooc_graph = nx.Graph()
            if filter_is_none:
                cooc_graph = self._build_cooc_graph(
                    type1, type2, None, cooc_graph, stopwords=stopwords
                )
            else:
                for doc_id in doc_ids:
                    cooc_graph = self._build_cooc_graph(
                        type1, type2, doc_id, cooc_graph, stopwords=stopwords
                    )
            yield cooc_graph, None

    def _build_cooc_graph(
        self, type1, type2, document, external_graph=None, stopwords={}
    ):
        """
        Co-occurrence graph constructor.
        This is a private method.
        """
        big_graph = self.big_graph.copy()
        print("Building co-ocurrence graph for types", type1, type2)
        print("The document is", document)
        print(
            "Number of nodes and edges in big graph are",
            big_graph.number_of_nodes(),
            big_graph.number_of_edges(),
        )

        # Retain sub-graph corresponding to nodes that appear in the document
        keep_nodes = []
        for node in big_graph.nodes:
            if node in keep_nodes:
                continue
            next_node = False
            for ent_type, list_words in stopwords.items():
                if big_graph.nodes[node]["type"] == ent_type:
                    if node in list_words:
                        next_node = True
            if next_node:
                continue

            if document is None:
                keep_nodes.append(node)
            else:            
                for appeareance in big_graph.nodes[node]["appears_in"]:
                    if document in appeareance["doc_id"]:
                        keep_nodes.append(node)
                        break
        print("A total of", len(keep_nodes), "nodes will be kept")
        doc_graph = nx.Graph(big_graph.subgraph(keep_nodes))
        keep_edges = []
        print(
            "Document graph has",
            doc_graph.number_of_nodes(),
            "nodes and",
            doc_graph.number_of_edges(),
            "edges",
        )

        for n1, n2 in doc_graph.edges:
            type_n1 = doc_graph.nodes[n1]["type"]
            type_n2 = doc_graph.nodes[n2]["type"]
            if (type1 == type_n1 and type2 == type_n2) or (
                type1 == type_n2 and type2 == type_n1
            ):
                keep_edges.append((n1, n2))
        print("A total of", len(keep_edges), "edges are kept")
        cooc_graph = nx.Graph(doc_graph.edge_subgraph(keep_edges))
        print(
            "Cooc graph has",
            cooc_graph.number_of_nodes(),
            "nodes and",
            cooc_graph.number_of_edges(),
            "edges",
        )

        # We now clean the edges and nodes appears_in list.
        if document is not None:
            for node in cooc_graph.nodes:
                appears_in = cooc_graph.nodes[node]["appears_in"]
                cooc_graph.nodes[node]["appears_in"] = [
                    d for d in appears_in if document in d["doc_id"]
                ]
            for u, v in cooc_graph.edges:
                appears_in = cooc_graph[u][v]["appears_in"]
                cooc_graph[u][v]["appears_in"] = [
                    d for d in appears_in if document in d["doc_id"]
                ]

        if external_graph is not None:
            print(
                "There is an external graph with",
                external_graph.number_of_edges(),
                "edges, and",
                external_graph.number_of_nodes(),
                "nodes",
            )
            for node, data in cooc_graph.nodes(data=True):
                if external_graph.has_node(node):
                    external_graph.nodes[node]["appears_in"].extend(data["appears_in"])
                else:
                    external_graph.add_node(node, **data)
            for n1, n2, data in cooc_graph.edges(data=True):
                if external_graph.has_edge(n1, n2):
                    external_graph[n1][n2]["appears_in"].extend(data["appears_in"])
                else:
                    external_graph.add_edge(n1, n2, **data)
            cooc_graph = external_graph
        for n1, n2, data in cooc_graph.edges(data=True):
            cooc_graph[n1][n2]["weight"] = len(data["appears_in"])
        print(
            "Resulting cooc graph has",
            cooc_graph.number_of_nodes(),
            "nodes and",
            cooc_graph.number_of_edges(),
            "edges",
        )
        return cooc_graph

    @typechecked
    def generate_cooccurrence_matrix(
        self,
        type_row: str,
        type_column: str,
        for_each_doc: bool = True,
        documents_filter: List[str] = None,
        graphs: List[Tuple[nx.Graph, Any]] = None,
        stopwords: Dict[str, List[str]] = {},
    ):
        """
        Builds a co-occurrence matrix of text features found on the
        same sentences.

        Parameters
        ----------
        type_row : str
            Features have types (see `BuildSource.build_big_table`'s
            documentation). Rows of the co-occurrence matrix will
            match a specific type.
        type_column : str
            Same as `type_row` but for the columns of the co-occurrence
            matrix.
        for_each_doc : bool
            A flag that indicates whether the method will return just
            one large co-ocurrence matrix corresponding to all the
            documents found in `self.parsed_documents_ids` (with some
            possible filters, see `documents_filter` argument), or
            a co-ocurrence matrix will be build for each document
            found in `self.parsed_documents_ids`.
        documents_filter : List[str]
            A list of document ids to filter documents.
        graphs : List[networkx.Graph]
            A list of graphs, if this one is given, `for_each_doc` and
            `documents_filter` are ignored. And the graphs will be used
            to build cooccurrence matrices.
        stopwords : Dict[str, List[str]]
            See documentation of BuildSource.parse_documents

        Yields
        ------
        numpy.ndarray
            A numpy array that has the cooccurrence matrix.
        str
            A document id
        """
        if graphs is None:
            graphs = self.generate_cooccurrence_graph(
                type_row,
                type_column,
                for_each_doc,
                documents_filter,
                stopwords=stopwords,
            )
        for graph, doc_id in graphs:
            yield self._build_cooc_matrix_from_graph(
                type_row,
                type_column,
                graph,
            ), doc_id

    @staticmethod
    def _build_cooc_matrix_from_graph(type_row, type_column, graph):
        # Get sizes
        row_nodes = [n for n in graph.nodes if graph.nodes[n]["type"] == type_row]
        column_nodes = [n for n in graph.nodes if graph.nodes[n]["type"] == type_column]

        matrix = np.zeros((len(row_nodes), len(column_nodes)))
        for i, row in enumerate(row_nodes):
            for j, col in enumerate(column_nodes):
                if (row, col) in graph.edges:
                    matrix[i, j] = graph[row][col]["weight"]
        return pd.DataFrame(data=matrix, columns=column_nodes, index=row_nodes)

    @typechecked
    def draw_frequent_words(
        self,
        type_filter: str,
        path: str,
        for_each_doc: bool = True,
        documents_filter: List[str] = None,
        show: bool = False,
        topn: int = 10,
        suffix: str = None,
        plotly: bool = True,
        stopwords: Dict[str, List[str]] = {},
    ):
        """
        Draws one or several frequency countplots from the corpus.

        Parameters
        ----------
        type_filter : str
            Select one type of linguistic feature (verb, adjective,
            person, organisation, etc.)
        path : str
            Path where a file will be stored, i.e. a file path.
        for_each_doc : bool
            If `True`, one plot will be created for each document
            that has been parsed. Else, all documents will be
            aggregated, and only one plot will be created for all
            the corpus.
        documents_filter : List[str], optional
            If `None`, all the corpus is used. Else, this is a list
            of strings, and all filepaths that were parsed and contain
            one of the given strings, will form a subcorpus to be used.
        show : bool
            Whether matplotlib plots are shown.
        topn : int
            Maximum number of linguistic features to show in the
            frequency counts.
        suffix : str, optional
            If `for_each_doc` is `True`, then path used to save each
            plot will be the name of the file and this suffix, saved
            in the folder referenced in `path`.
        plotly: bool
            Whether the figures will be created with plotly (True)
            or with seaborn (False)
        stopwords : Dict[str, List[str]]
            See documentation of BuildSource.parse_documents
        """
        df = self.big_table[self.big_table.type == type_filter]
        if df.empty:
            return None
        if type_filter in stopwords.keys():
            df = df[
                df.text.apply(lambda x: False if x in stopwords[type_filter] else True)
            ]
        if df.empty:
            return None
        if documents_filter is not None:
            doc_ids = []
            for doc_filter in documents_filter:
                for parsed_doc in self.parsed_documents_ids:
                    if doc_filter in parsed_doc:
                        doc_ids.append(parsed_doc)
                        break
        else:
            doc_ids = self.parsed_documents_ids

        if for_each_doc:
            if not os.path.isdir(path):
                dirpath = os.path.dirname(path)
            else:
                dirpath = path
            for doc_id in doc_ids:
                subdf = df[df.doc_id.str.contains(doc_id)]
                if not subdf.empty:
                    path_doc = os.path.join(
                        dirpath,
                        f"{''.join(doc_id.split('/')[-1].split('.')[:-1])}_{suffix}.pdf",
                    )
                    countdf = subdf.text.value_counts().reset_index(name="value")
                    countdf.to_csv(path_doc.replace(".pdf", ".csv"), index=False)
                    if plotly:
                        frequency_graph(
                            countdf,
                            path_doc.replace(".pdf", "_wordcloud.png"),
                            path_doc.replace(".pdf", ".html"),
                        )
                    else:
                        self._draw_countplot(subdf, path_doc, topn, show)
        else:
            subdf = df[df.doc_id.str.contains("|".join(doc_ids))]
            if not subdf.empty:
                countdf = subdf.text.value_counts().reset_index(name="value")
                countdf.to_csv(path.replace(".pdf", ".csv"), index=False)
                if plotly:
                    frequency_graph(
                        countdf,
                        path.replace(".pdf", "_wordcloud.png"),
                        path.replace(".pdf", ".html"),
                    )
                else:
                    self._draw_countplot(subdf, path, topn, show)

    @staticmethod
    def _draw_countplot(
        df: pd.DataFrame, path: str, topn: int = 35, show: bool = False
    ):
        fig, ax = plt.subplots()
        uniques = df.text.unique()
        truncation = min([topn, len(uniques)])
        palette = sns.cubehelix_palette(
            start=2, rot=0, dark=0.2, light=0.7, reverse=True, n_colors=topn
        )
        sns.countplot(
            y="text",
            data=df,
            order=pd.value_counts(df.text).iloc[:truncation].index,
            palette=palette,
            ax=ax,
        )
        plt.savefig(path, dpi=300, bbox_inches="tight")
        if show:
            plt.tight_layout()
            plt.show()
        plt.close(fig)

    @typechecked
    def draw_heatmaps(
        self,
        type_row: str,
        type_column: str,
        path: str,
        for_each_doc: bool = True,
        documents_filter: List[str] = None,
        graphs: List[Tuple[nx.Graph, Any]] = None,
        show: bool = False,
        topn: int = 10,
        suffix: str = None,
        plotly: bool = True,
        stopwords: Dict[str, List[str]] = {},
    ):
        """
        # TODO add documentation
        # here
        """
        palette = sns.cubehelix_palette(
            start=2, rot=0, dark=0.2, light=1.0, reverse=False, as_cmap=True
        )
        matrices = self.generate_cooccurrence_matrix(
            type_row, type_column, for_each_doc, documents_filter, graphs, stopwords
        )
        for matrix, doc_id in matrices:
            if matrix.empty:
                continue
            if doc_id is not None:
                if not os.path.isdir(path):
                    dirpath = os.path.dirname(path)
                else:
                    dirpath = path
                path_doc = os.path.join(
                    dirpath,
                    f"{''.join(doc_id.split('/')[-1].split('.')[:-1])}_{suffix}.pdf",
                )
            else:
                path_doc = path
            matrix.to_csv(path_doc.replace(".pdf", ".csv"), index=False)
            nrows, ncols = matrix.shape
            row_idx = (
                matrix.sum(axis="columns")
                .sort_values(ascending=False)
                .index[: min([topn, nrows])]
            )
            matrix = matrix.loc[row_idx, :]
            col_idx = (
                matrix.sum(axis="index")
                .sort_values(ascending=False)
                .index[: min([topn, ncols])]
            )
            matrix = matrix.loc[:, col_idx]

            # Rename features
            new_rows = {old: old.split("_")[0] for old in matrix.index}
            new_cols = {old: old.split("_")[0] for old in matrix.columns}
            matrix = matrix.rename(columns=new_cols, index=new_rows)
            if plotly:
                valx = matrix.columns.values
                valy = matrix.index.values
                heatMap(
                    valx,
                    valy,
                    type_column,
                    type_row,
                    matrix.applymap(lambda x: np.log(x + 1)).values,
                    path_doc.replace(".pdf", ".html"),
                )
            else:
                sns.heatmap(matrix.applymap(lambda x: np.log(x + 1)), cmap=palette)
                plt.savefig(path_doc, dpi=300, bbox_inches="tight")
                if show:
                    plt.tight_layout()
                    plt.show()
                plt.close()

    @typechecked
    def draw_graphs(
        self,
        type_one: str,
        type_two: str,
        path: str,
        for_each_doc: bool = True,
        documents_filter: List[str] = None,
        graphs: List[Tuple[nx.Graph, Any]] = None,
        max_conns: int = 15,
        height: int = 600,
        width: int = 800,
        heading: str = None,
        suffix: str = None,
        plotly: bool = True,
        stopwords: Dict[str, List[str]] = {},
    ):
        """
        # TODO add documentation
        # here
        """
        if graphs is None:
            graphs = self.generate_cooccurrence_graph(
                type_one, type_two, for_each_doc, documents_filter, stopwords
            )
        for graph, doc_id in graphs:
            graph_raw = graph.copy()
            sorted_edges = sorted(
                graph.edges(data=True), key=lambda t: t[2].get("weight")
            )
            truncation = -min([max_conns, len(sorted_edges)])
            keep_edges = [(e[0], e[1]) for e in sorted_edges[truncation:]]
            print("Graph corresponding to id", doc_id)
            print("Number of edges", graph.number_of_edges())
            print("Number of nodes", graph.number_of_nodes())
            graph = nx.Graph(graph.edge_subgraph(keep_edges))
            print("After pruning")
            print("Number of edges", graph.number_of_edges())
            print("Number of nodes", graph.number_of_nodes())
            if graph.number_of_edges() == 0:
                continue
            for node in graph.nodes:
                size = len(graph.nodes[node]["appears_in"])
                graph.nodes[node]["size"] = size
                del graph.nodes[node]["appears_in"]
            for u, v in graph.edges:
                del graph[u][v]["appears_in"]
            print("After pruning")
            print("Number of edges", graph.number_of_edges())
            print("Number of nodes", graph.number_of_nodes())
            print("Digitising...")
            self._digitise_weights(graph)
            if doc_id is not None:
                if not os.path.isdir(path):
                    dirpath = os.path.dirname(path)
                else:
                    dirpath = path
                path_doc = os.path.join(
                    dirpath,
                    f"{''.join(doc_id.split('/')[-1].split('.')[:-1])}_{suffix}.html",
                )
            else:
                path_doc = path
            if plotly:
                net(graph, type_one, type_two, path_doc)
            else:
                nt = Network(f"{height}px", f"{width}px", heading=heading)
                nx2pyvis(graph, nt)
                print("Trying to save in path", path_doc)
                nt.save_graph(path_doc)
            nx.write_pajek(graph_raw, path_doc.replace(".html", ".net"))

    @staticmethod
    def _digitise_weights(
        graph,
        size_key="size",
        weight_key="weight",
        size_factor=5,
        weight_factor=5,
        quantiles=[0, 0.25, 0.5, 0.75, 1.0],
    ):
        edges = graph.edges(data=True)
        nodes = graph.nodes(data=True)
        for e in edges:
            print(e)
            break
        widths = [e[2][weight_key] for e in edges]
        try:
            bins1 = [np.quantile(widths, q) for q in quantiles]
            widths_digits = (np.digitize(widths, bins1) + 2) * weight_factor
            sizes = [n[1][size_key] for n in nodes]
            bins = [np.quantile(sizes, q) for q in quantiles]
            sizes_digits = np.digitize(sizes, bins) * size_factor
        except Exception as e:
            print("widths", widths)
            print("bins_widths", bins1)
            print("sizes", sizes)
            print("bins_sizes", bins)
            raise e
        for i, (node, data) in enumerate(nodes):
            graph.nodes[node]["size"] = int(sizes_digits[i])
        for i, (u, v, data) in enumerate(edges):
            graph[u][v]["weight"] = int(widths_digits[i])


if __name__ == "__main__":
    from load_source import LoadSource

    path = "../../data/input/tierras"
    filename = "5326_sentencia Drummond Los Naranjos.pdf"
    ls = LoadSource(
        db_name="sentences_tierras",
        db_coll="document",
        path=path,
        filter_file_names=[filename],
    )
    ls.load_files_tierras()
    bs = BuildSource(ls, spacy_model_name="es_core_news_sm")
    bs.parse_documents({filename: [1, 2]})
    bs.build_big_table()
    print(bs.big_table)
    bs.build_big_graph()
    cooc = list(
        bs.generate_cooccurrence_graph(
            "adjective", "adjective", documents_filter=["Los Naranjos"]
        )
    )
    matgen = list(bs.generate_cooccurrence_matrix("adjective", "adjective"))
    bs.draw_frequent_words("adjective", path="./myfigure.pdf", show=True)
    bs.draw_heatmaps("adjective", "adjective", "./myheatmap.pdf", show=True)
    bs.draw_graphs(
        "adjective", "adjective", "./mynetwork.html", heading="adj-adj", graphs=cooc
    )
