import glob
import pdftotext
from datetime import datetime
import spacy
import re
import nltk
import sys
from tqdm import tqdm
from time import sleep
import matplotlib.pyplot as plt
from collections import Counter
import re
import csv
import networkx as nx
import os
import pandas as pd
import seaborn as sns
import random
import graphistry
import pickle
from chart_studio.plotly import plot as py, iplot
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

ROOT_PATH = os.path.dirname(os.sep.join(os.path.abspath(__file__).split(os.sep)[:-3]))
sys.path.insert(1, ROOT_PATH)

from src.corpus.load_source import LoadSource
from src.util.common import Common
from src.util.data_acces import DataAccessMongo
from root import DIR_OUTPUT

#graphistry.register(api=3, username='morenoluis', password='xyz')
class BuildSource(object):
    def __init__(self, **kwargs):
        print("Start process BuildSource")
        self.lfo = kwargs['load_files_object']
        self.filter_pages = kwargs.get('filter_pages', {})
        self.name_global_model = "es_core_news_md"
        self.nlp = spacy.load(self.name_global_model)
        self.verb_pattern = [{"POS": "VERB", "OP": "*"}, {"POS": "ADV", "OP": "*"}, {"POS": "VERB", "OP": "+"},
                             {"POS": "PART", "OP": "*"}]
        # Label for each by sent
        self.label_sent = []
        self.label_file = {}
        # Labels parser
        self.label_chunk_sent = {}
        self.label_chunk = []
        self.label_labels_sent = {}
        self.label_labels = []
        self.label_per_sent = {}
        self.label_per = []
        self.label_org_sent = {}
        self.label_org = []
        self.label_verb_sent = {}
        self.label_verb = []
        self.label_adj_sent = {}
        self.label_adj = []
        # Maps parser
        self.heat_per = {}
        self.heat_org = {}
        self.heat_per_org = {}
        self.heat_verb_per = {}
        self.heat_verb_org = {}
        self.heat_adj_per = {}
        self.heat_adj_org = {}
        self.heat_per_x = []
        self.heat_per_y = []
        self.heat_org_x = []
        self.heat_org_y = []
        self.heat_per_org_x = []
        self.heat_per_org_y = []
        self.heat_verb_per_x = []
        self.heat_verb_per_y = []
        self.heat_verb_org_x = []
        self.heat_verb_org_y = []
        self.heat_adj_org_x = []
        self.heat_adj_org_y = []
        self.heat_adj_per_x = []
        self.heat_adj_per_y = []

    def parser_document(self):
        for i in tqdm(range(0, len(self.lfo.list_texts_corpus)), desc="Structure Corpus"):
            single_doc = self.lfo.list_texts_corpus[i]
            count_page = 0
            count_sent = 0
            self.lfo.list_texts_corpus[i]['doc_parser'] = []
            try:
                if len(self.filter_pages) > 0:
                    pages_raw = [(item_p, single_doc['pages_raw'][item_p - 1]) for item_k in self.filter_pages[i]
                                 for item_p in self.filter_pages[i][item_k]]
                else:
                    pages_raw = list(enumerate(single_doc['pages_raw']))
            except Exception as e:
                print(e)
                pages_raw = list(enumerate(single_doc['pages_raw']))
            print('filter pages: ', self.filter_pages, 'count: ', len(pages_raw))
            for idx_p, page_raw in pages_raw:
                count_page += 1
                count_sent_local = 0
                text_raw = page_raw
                text_raw = re.sub('\n+', ' ', text_raw)
                text_raw = re.sub('\s+', ' ', text_raw)
                sentences_raw = nltk.sent_tokenize(text_raw)
                sentences_parser = []
                for item in sentences_raw:
                    dict_sent = {'page': idx_p, 'page_count': count_page, 'local_sent': count_sent_local,
                                 'global_sent': count_sent,
                                 'text_raw': item, 'text_clear': '', 'chunks': [], 'verbs': [], 'entities': [],
                                 'adj': [], 'noun_adj_pairs': {}, 'labels': []}
                    count_sent += 1
                    count_sent_local += 1

                    result_label_extract = re.findall(r'(?<=\[)(.*?)(?=\])', item)
                    labels_result = []
                    for item_label in result_label_extract:
                        key = ''
                        item_label = re.sub(r"[^- \w]|_", "", item_label)
                        value = item_label
                        for word in item_label.split(' '):
                            # print(word)
                            if word.isupper():
                                # print(word)
                                key = word
                                break
                        for word in item_label.split(' '):
                            # print(word)
                            if word.isnumeric():
                                # print(word)
                                value.replace(word, '')
                        # value = re.sub(r"[^- \w]|_", "", item_label)
                        value = value.replace(key, '').lstrip()
                        if key == '':
                            key = item.upper()
                        if value == '':
                            value = key
                        labels_result.append({'type': key, 'text': value})
                    dict_sent['labels'] = labels_result

                    item = re.sub(r'(?<=\[)(.*?)(?=\])', "", item)
                    item = item.replace('ENT:', '').replace('ENT2:', '').replace('TEST:', '').replace('TEST2:', '').replace('[]', '')
                    item = re.sub(r"[^- \w]|_", "", item)

                    dict_sent['text_clear'] = item
                    doc = self.nlp(item)
                    count_chunk = 0
                    # print('sent # ', count_sent, '-' * 150)
                    # print('\t with process:', item)

                    for np in doc.noun_chunks:
                        count_chunk += 1
                        dict_sent['chunks'].append({'num': count_chunk, 'text': Common.clean_text_all(np.text)})
                    for entity in doc.ents:
                        dict_sent['entities'].append({'type': entity.label_, 'text': Common.clean_text_all(entity.text)})
                    # nlp_verb_phrases = textacy.make_spacy_doc(item, lang=self.name_global_model)
                    # verb_phrases = textacy.extract.matches(nlp_verb_phrases, self.verb_pattern)
                    # count_chunk = 0
                    # for vp in verb_phrases:
                    #     count_chunk += 1
                    #     dict_sent['verbs'].append({'num': count_chunk, 'text': Common.clean_text_all(vp.text)})

                    noun_adj_pairs = {}
                    all_adj = []
                    count_chunk = 0
                    count_verb = 0
                    for tok in doc:
                        adj = []
                        noun = ""
                        # for tok in token:
                        if tok.pos_ == "NOUN":
                            noun = tok.lemma_
                        if tok.pos_ == "ADJ":
                            count_chunk += 1
                            adj.append(tok.lemma_)
                            dict_sent['adj'].append({'num': count_chunk, 'text': Common.clean_text_all(tok.lemma_)})
                        if tok.pos_ == "VERB":
                            count_verb += 1
                            dict_sent['verbs'].append({'num': count_chunk, 'text': Common.clean_text_all(tok.lemma_)})
                        if noun:
                            noun_adj_pairs.update({noun: adj})
                    dict_sent['noun_adj_pairs'] = noun_adj_pairs
                    # expected output
                    # print(dict_sent['adj'])
                    self.lfo.list_texts_corpus[i]['sentences'].append(dict_sent)
                    sentences_parser.append(dict_sent)
                self.lfo.list_texts_corpus[i]['doc_parser'].append({'page': idx_p, 'sentences': sentences_parser})

    def print_sentences_parser(self):
        count = 0
        for item in self.lfo.list_texts_corpus:
            count += 1
            print('-> ', count, '\tfile_name:', item['file_name'][:50].zfill(50).replace('0', ' '),
                  item['doc_parser'][0])

    def analyzer_doc(self):
        for item in self.lfo.list_texts_corpus:
            key_name = Common.clean_text_all(item['file_name'])
            self.label_sent.append(key_name)
            self.label_file[key_name] = item
            print('\tanalyzer_doc :', key_name)
            self.label_chunk_sent[key_name] = [item_chunk['text'].lower() for sent in item['sentences'] for item_chunk
                                               in
                                               sent['chunks'] if len(item_chunk['text'].split(' ')) > 1 and len(
                    item_chunk['text'].split(' ')) < 5]
            self.label_chunk += self.label_chunk_sent[key_name]

            self.label_labels_sent[key_name] = [item_chunk['text'].lower() for sent in item['sentences'] for item_chunk
                                               in sent['labels']]
            self.label_labels += self.label_labels_sent[key_name]

            self.label_per_sent[key_name] = [item_ent['text'].lower() for sent in item['sentences'] for item_ent in
                                             sent['entities'] if item_ent['type'] in ['PER'] and len(
                    item_ent['text'].split(' ')) > 1 and len(item_ent['text'].split(' ')) < 5]
            self.label_per += self.label_per_sent[key_name]
            self.label_org_sent[key_name] = [item_ent['text'].lower() for sent in item['sentences'] for item_ent in
                                             sent['entities'] if item_ent['type'] in ['ORG'] and len(
                    item_ent['text'].split(' ')) > 1 and len(item_ent['text'].split(' ')) < 5]
            self.label_org += self.label_org_sent[key_name]
            self.label_verb_sent[key_name] = [item_verb['text'].lower() for sent in item['sentences'] for item_verb in
                                              sent['verbs']]  # if item_ent['type'] == 'PER'
            self.label_verb += self.label_verb_sent[key_name]
            self.label_adj_sent[key_name] = [item_adj['text'].lower() for sent in item['sentences'] for item_adj in
                                             sent['adj']]  # if item_ent['type'] == 'PER'
            self.label_adj += self.label_adj_sent[key_name]

            heat_per_x = []
            heat_per_y = []
            heat_org_x = []
            heat_org_y = []
            heat_per_org_x = []
            heat_per_org_y = []
            heat_verb_per_x = []
            heat_verb_per_y = []
            heat_verb_org_x = []
            heat_verb_org_y = []
            heat_adj_org_x = []
            heat_adj_org_y = []
            heat_adj_per_x = []
            heat_adj_per_y = []

            for sent in item['sentences']:
                for value_x in sent['entities']:
                    for value_y in sent['entities']:
                        if value_x != value_y and value_x['type'] in ['PER'] and len(
                                value_x['text'].split(' ')) > 1 and len(value_x['text'].split(' ')) < 5:  # 'PER', 'ORG'
                            if value_y['type'] in ['PER'] and len(value_y['text'].split(' ')) > 1 and len(
                                    value_y['text'].split(' ')) < 5:  # 'PER', 'ORG'
                                heat_per_x.append(value_x['text'].lower())
                                heat_per_y.append(value_y['text'].lower())
                            if value_y['type'] in ['ORG'] and len(value_y['text'].split(' ')) > 1 and len(
                                    value_y['text'].split(' ')) < 5:  # 'PER', 'ORG'
                                heat_per_org_x.append(value_x['text'].lower())
                                heat_per_org_y.append(value_y['text'].lower())
                        if value_x != value_y and value_x['type'] in ['ORG'] and len(value_x['text'].split(' ')) < 5:
                            if value_y['type'] in ['ORG'] and len(value_y['text'].split(' ')) > 1 and len(
                                    value_y['text'].split(' ')) < 5:  # 'PER', 'ORG'
                                heat_org_x.append(value_x['text'].lower())
                                heat_org_y.append(value_y['text'].lower())
                    for value_y in sent['verbs']:
                        if value_x != value_y and value_x['type'] in ['PER'] and len(
                                value_x['text'].split(' ')) > 1 and len(value_x['text'].split(' ')) < 5:  # 'PER', 'ORG'
                            heat_verb_per_x.append(value_x['text'].lower())
                            heat_verb_per_y.append(value_y['text'].lower())
                        if value_x != value_y and value_x['type'] in ['ORG'] and len(
                                value_x['text'].split(' ')) > 1 and len(value_x['text'].split(' ')) < 5:  # 'PER', 'ORG'
                            heat_verb_org_x.append(value_x['text'].lower())
                            heat_verb_org_y.append(value_y['text'].lower())
                    for value_y in sent['adj']:
                        if value_x != value_y and value_x['type'] in ['PER'] and len(
                                value_x['text'].split(' ')) > 1 and len(value_x['text'].split(' ')) < 5:  # 'PER', 'ORG'
                            heat_adj_per_x.append(value_x['text'].lower())
                            heat_adj_per_y.append(value_y['text'].lower())
                        if value_x != value_y and value_x['type'] in ['ORG'] and len(
                                value_x['text'].split(' ')) > 1 and len(value_x['text'].split(' ')) < 5:  # 'PER', 'ORG'
                            heat_adj_org_x.append(value_x['text'].lower())
                            heat_adj_org_y.append(value_y['text'].lower())
                for value_x in sent['labels']:
                    pass

            self.heat_per_x += heat_per_x
            self.heat_per_y += heat_per_y
            self.heat_org_x += heat_org_x
            self.heat_org_y += heat_org_y
            self.heat_per_org_x += heat_per_org_x
            self.heat_per_org_y += heat_per_org_y
            self.heat_verb_per_x += heat_verb_per_x
            self.heat_verb_per_y += heat_verb_per_y
            self.heat_verb_org_x += heat_verb_org_x
            self.heat_verb_org_y += heat_verb_org_y
            self.heat_adj_org_x += heat_adj_org_x
            self.heat_adj_org_y += heat_adj_org_y
            self.heat_adj_per_x += heat_adj_per_x
            self.heat_adj_per_y += heat_adj_per_y

            self.heat_per[key_name] = {'x': heat_per_x, 'y': heat_per_y}
            self.heat_org[key_name] = {'x': heat_org_x, 'y': heat_org_y}
            self.heat_per_org[key_name] = {'x': heat_per_org_x, 'y': heat_per_org_y}
            self.heat_verb_per[key_name] = {'x': heat_verb_per_x, 'y': heat_verb_per_y}
            self.heat_verb_org[key_name] = {'x': heat_verb_org_x, 'y': heat_verb_org_y}
            self.heat_adj_per[key_name] = {'x': heat_adj_per_x, 'y': heat_adj_per_y}
            self.heat_adj_org[key_name] = {'x': heat_adj_org_x, 'y': heat_adj_org_y}

        print('\t\tlabel_chunk', len(self.label_chunk), len(set(self.label_chunk)))
        print('\t\tlabel_per', len(self.label_per), len(set(self.label_per)))
        print('\t\tlabel_org', len(self.label_org), len(set(self.label_org)))
        print('\t\tlabel_verb', len(self.label_verb), len(set(self.label_verb)))
        print('\t\theat_per_x', len(self.heat_per_x), len(set(self.heat_per_x)))
        print('\t\theat_org_x', len(self.heat_org_x), len(set(self.heat_org_x)))
        print('\t\theat_per_org_x', len(self.heat_per_org_x), len(set(self.heat_per_org_x)))
        print('\t\theat_verb_per_x', len(self.heat_verb_per_x), len(set(self.heat_verb_per_x)))
        print('\t\theat_verb_org_x', len(self.heat_verb_org_x), len(set(self.heat_verb_org_x)))
        print('\t\theat_adj_per_x', len(self.heat_adj_per_x), len(set(self.heat_adj_per_x)))
        print('\t\theat_adj_org_x', len(self.heat_adj_org_x), len(set(self.heat_adj_org_x)))

    def graph_docs(self):
        dict_report = [{'label': 'labels', 'item': self.label_labels, 'sent': self.label_labels_sent},
                       {'label': 'chunk', 'item': self.label_chunk, 'sent': self.label_chunk_sent},
                       {'label': 'entities_person', 'item': self.label_per, 'sent': self.label_per_sent},
                       {'label': 'entities_org', 'item': self.label_org, 'sent': self.label_org_sent},
                       {'label': 'verb', 'item': self.label_verb, 'sent': self.label_verb_sent},
                       {'label': 'adj', 'item': self.label_adj, 'sent': self.label_adj_sent}]

        folder_name = "{}/{}/{}/".format(DIR_OUTPUT, self.lfo.db_name, datetime.now().strftime('%Y-%m-%d'))
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        date_file_name = "{}/{}".format(folder_name, datetime.now().strftime('%Y-%m-%d')) # %Y-%m-%d_%H-%M

        # print('date_file_name', date_file_name)

        for item_report in dict_report:
            word_freq = Counter(item_report['item'])
            common_words = word_freq.most_common(100)
            # print('label_sent', self.label_sent)
            # print('common_words', common_words)
            # print('item_report', item_report)
            # print('item_report[sent]', item_report['sent'])
            cfd = nltk.ConditionalFreqDist(
                (title, word)
                for title in self.label_sent
                for word in item_report['sent'][title])

            label_words = [word for (word, freq) in common_words]
            # print(label_words)
            try:
                fig = plt.figure(figsize=(20, 8))
                plt.gcf().subplots_adjust(bottom=0.25)
                cfd.plot(title="Most common {}".format(item_report['label']), conditions=self.label_sent, samples=label_words)
                plt.show()
                fig.savefig('{}_all_{}_img_1_cont.png'.format(date_file_name, item_report['label']), bbox_inches="tight")

                fig = plt.figure(figsize=(20, 8))
                plt.gcf().subplots_adjust(bottom=0.25)  # to avoid x-ticks cut-off
                cfd.plot(title="Most common {} accumulated".format(item_report['label']), conditions=self.label_sent, samples=label_words, cumulative=True)
                plt.show()
                fig.savefig('{}_all_{}_img_2_cum.png'.format(date_file_name, item_report['label']), bbox_inches="tight")
            except Exception as e:
                print(e)


            with open('{}_all_{}_freq.csv'.format(date_file_name, item_report['label']), 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["item", "value"])
                for item, value in word_freq.items():
                    writer.writerow([item, value])

    @staticmethod
    def compare_docs_report(dict_report, date_file_name):
        for item_report in dict_report:
            # print(item_report)
            x = item_report['x']
            y = item_report['y']
            if len(x) > 0 and len(y) > 0:
                df = pd.DataFrame([x, y]).T
                df.columns = ['Nombres', 'Relaciones']
                df['count'] = 1
                df2 = df.groupby(['Nombres', 'Relaciones'], as_index=False).count()
                df_p = pd.pivot_table(df2, 'count', 'Relaciones', 'Nombres')
                df_p.to_csv(date_file_name + '_heat_{}.csv'.format(item_report['label']), index=True)
                #ax = plt.axes()
                plt.figure(figsize=(25, 20))
                ax = plt.axes()
                sns_plot = sns.heatmap(df_p)
                ax.set_title(item_report['title'])
                sns_plot.figure.savefig(date_file_name + '_heat_{}.png'.format(item_report['label']))

    def compare_docs(self, **kwargs):
        all_docs = kwargs.get('all_docs', False)
        if all_docs:
            dict_report = [{'label': 'per', 'x': [], 'y': [], 'title': 'Personas'},
                           {'label': 'org', 'x': [], 'y': [], 'title': 'Organizaciones'},
                           {'label': 'per_verb', 'x': [], 'y': [], 'title': 'Personas vs Acciones'},
                           {'label': 'org_verb', 'x': [], 'y': [], 'title': 'Organizaciones vs Acciones'},
                           {'label': 'per_adj', 'x': [], 'y': [], 'title': 'Personas vs Adjetivos'},
                           {'label': 'org_adj', 'x': [], 'y': [], 'title': 'Organizaciones vs Adjetivos'},
                           {'label': 'per_org', 'x': [], 'y': [], 'title': 'Persona y Organizaciones'}]
            for key_name in self.label_sent:
                dict_report[0]['x'] += self.heat_per[key_name]['x']
                dict_report[0]['y'] += self.heat_per[key_name]['y']
                dict_report[1]['x'] += self.heat_org[key_name]['x']
                dict_report[1]['y'] += self.heat_org[key_name]['y']
                dict_report[2]['x'] += self.heat_verb_per[key_name]['x']
                dict_report[2]['y'] += self.heat_verb_per[key_name]['y']
                dict_report[3]['x'] += self.heat_verb_org[key_name]['x']
                dict_report[3]['y'] += self.heat_verb_org[key_name]['y']
                dict_report[4]['x'] += self.heat_adj_per[key_name]['x']
                dict_report[4]['y'] += self.heat_adj_per[key_name]['y']
                dict_report[5]['x'] += self.heat_adj_org[key_name]['x']
                dict_report[5]['y'] += self.heat_adj_org[key_name]['y']
                dict_report[6]['x'] += self.heat_per_org[key_name]['x']
                dict_report[6]['y'] += self.heat_per_org[key_name]['y']

            folder_name = "{}/{}/{}/".format(DIR_OUTPUT, self.lfo.db_name, datetime.now().strftime('%Y-%m-%d'))
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
            date_file_name = "{}/{}_all".format(folder_name, datetime.now().strftime('%Y-%m-%d')) # %Y-%m-%d_%H-%M
            self.compare_docs_report(dict_report, date_file_name)

        else:
            for key_name in self.label_sent:
                dict_report = [{'label': 'per', 'x': self.heat_per[key_name]['x'], 'y': self.heat_per[key_name]['y']},
                               {'label': 'org', 'x': self.heat_org[key_name]['x'], 'y': self.heat_org[key_name]['y']},
                               {'label': 'per_verb', 'x': self.heat_verb_per[key_name]['x'], 'y': self.heat_verb_per[key_name]['y']},
                               {'label': 'org_verb', 'x': self.heat_verb_org[key_name]['x'], 'y': self.heat_verb_org[key_name]['y']},
                               {'label': 'per_adj', 'x': self.heat_adj_per[key_name]['x'], 'y': self.heat_adj_per[key_name]['y']},
                               {'label': 'org_adj', 'x': self.heat_adj_org[key_name]['x'], 'y': self.heat_adj_org[key_name]['y']},
                               {'label': 'per_org', 'x': self.heat_per_org[key_name]['x'], 'y': self.heat_per_org[key_name]['y']}]

                folder_name = "{}/{}/{}/".format(DIR_OUTPUT, self.lfo.db_name, datetime.now().strftime('%Y-%m-%d'))
                if not os.path.exists(folder_name):
                    os.makedirs(folder_name)
                date_file_name = "{}/{}-{}".format(folder_name, datetime.now().strftime('%Y-%m-%d'), key_name) # %Y-%m-%d_%H-%M
                self.compare_docs_report(dict_report, date_file_name)

    @staticmethod
    def network_docs_report(dict_network, date_file_name, folder_name):
        # plt.figure(figsize=(10, 5))
        # ax = plt.gca()
        # ax.set_title('Random graph')
        # G = nx.fast_gnp_random_graph(10, 0.2)
        # nx.draw(G, with_labels=True, node_color='lightgreen', ax=ax)
        # _ = ax.axis('off')
        for item_report in dict_network:
            # print(item_report)
            d = {"source": item_report['x'], "target": item_report['y']}
            df = pd.DataFrame(d)
            g = nx.from_pandas_edgelist(df, 'source', 'target')
            fig = plt.figure(figsize=(20, 15))
            ax = plt.gca()
            ax.set_title(item_report['title'])
            nx.write_pajek(g, '{}_network_{}.net'.format(date_file_name, item_report['label']))
            nx.draw(g, with_labels=True, font_weight='bold')
            plt.show()
            fig.savefig('{}_network_{}.png'.format(date_file_name, item_report['label']), bbox_inches="tight")
            BuildSource.scatter_plot_2d(g, folder_name, item_report['title'])
            #g = graphistry.bind(source='source', destination='target').edges(df)
            #g.plot()

    @staticmethod
    def scatter_plot_2d(G, folderPath, name, savePng=False):
        print("Creating scatter plot (2D)...")

        Nodes = [comp for comp in nx.connected_components(G)]  # Looks for the graph's communities
        Edges = G.edges()
        edge_weights = nx.get_edge_attributes(G, 'weight')

        labels = []  # names of the nodes to plot
        group = []  # id of the communities
        group_cnt = 0

        print("Communities | Number of Nodes")
        for subgroup in Nodes:
            group_cnt += 1
            print("      %d     |      %d" % (group_cnt, len(subgroup)))
            for node in subgroup:
                labels.append(node)  # int(node)
                group.append(group_cnt)

        labels, group = (list(t) for t in zip(*sorted(zip(labels, group))))

        layt = nx.spring_layout(G, dim=2)  # Generates the layout of the graph
        Xn = [layt[k][0] for k in list(layt.keys())]  # x-coordinates of nodes
        Yn = [layt[k][1] for k in list(layt.keys())]  # y-coordinates
        Xe = []
        Ye = []

        plot_weights = []
        for e in Edges:
            Xe += [layt[e[0]][0], layt[e[1]][0], None]
            Ye += [layt[e[0]][1], layt[e[1]][1], None]
            ax = (layt[e[0]][0] + layt[e[1]][0]) / 2
            ay = (layt[e[0]][1] + layt[e[1]][1]) / 2
            # plot_weights.append((edge_weights[(e[0], e[1])], ax, ay))

        annotations_list = [
            dict(
                x=plot_weight[1],
                y=plot_weight[2],
                xref='x',
                yref='y',
                text=plot_weight[0],
                showarrow=True,
                arrowhead=7,
                ax=plot_weight[1],
                ay=plot_weight[2]
            )
            for plot_weight in plot_weights
        ]

        trace1 = go.Scatter(x=Xe,
                            y=Ye,
                            mode='lines',
                            line=dict(color='rgb(90, 90, 90)', width=1),
                            hoverinfo='none'
                            )

        trace2 = go.Scatter(x=Xn,
                            y=Yn,
                            mode='markers+text',
                            name='Nodes',
                            marker=dict(symbol='circle',
                                        size=8,
                                        color=group,
                                        colorscale='Viridis',
                                        line=dict(color='rgb(255,255,255)', width=1)
                                        ),
                            text=labels,
                            textposition='top center',
                            hoverinfo='none'
                            )

        xaxis = dict(
            backgroundcolor="rgb(200, 200, 230)",
            gridcolor="rgb(255, 255, 255)",
            showbackground=True,
            zerolinecolor="rgb(255, 255, 255)"
        )
        yaxis = dict(
            backgroundcolor="rgb(230, 200,230)",
            gridcolor="rgb(255, 255, 255)",
            showbackground=True,
            zerolinecolor="rgb(255, 255, 255)"
        )
        #
        #    width=700,
        #    height=700,
        layout = go.Layout(
            title=name,
            showlegend=False,
            plot_bgcolor="rgb(230, 230, 200)",
            scene=dict(
                xaxis=dict(xaxis),
                yaxis=dict(yaxis)
            ),
            margin=dict(
                t=100
            ),
            hovermode='closest',
            annotations=annotations_list
            , )
        data = [trace1, trace2]
        fig = go.Figure(data=data, layout=layout)
        plotDir = folderPath + "/"

        print("Plotting..")

        if savePng:
            plot(fig, filename=plotDir + name + ".html", auto_open=True, image='png', image_filename=plotDir + name,
                 output_type='file', image_width=700, image_height=700, validate=False)
        else:
            plot(fig, filename=plotDir + name + ".html")

    def network_docs(self, **kwargs):
        all_docs = kwargs.get('all_docs', False)
        if all_docs:
            dict_network = [{'label': 'per', 'x': [], 'y': [], 'title': 'Personas'},
                            {'label': 'org', 'x': [], 'y': [], 'title': 'Organizaciones'},
                            {'label': 'per_verb', 'x': [], 'y': [], 'title': 'Personas vs Acciones'},
                            {'label': 'org_verb', 'x': [], 'y': [], 'title': 'Organizaciones vs Acciones'},
                            {'label': 'per_adj', 'x': [], 'y': [], 'title': 'Personas vs Adjetivos'},
                            {'label': 'org_adj', 'x': [], 'y': [], 'title': 'Organizaciones vs Adjetivos'},
                            {'label': 'per_org', 'x': [], 'y': [], 'title': 'Persona y Organizaciones'}]

            for key_name in self.label_sent:
                dict_network[0]['x'] += self.heat_per[key_name]['x']
                dict_network[0]['y'] += self.heat_per[key_name]['y']
                dict_network[1]['x'] += self.heat_org[key_name]['x']
                dict_network[1]['y'] += self.heat_org[key_name]['y']
                dict_network[2]['x'] += self.heat_verb_per[key_name]['x']
                dict_network[2]['y'] += self.heat_verb_per[key_name]['y']
                dict_network[3]['x'] += self.heat_verb_org[key_name]['x']
                dict_network[3]['y'] += self.heat_verb_org[key_name]['y']
                dict_network[4]['x'] += self.heat_adj_per[key_name]['x']
                dict_network[4]['y'] += self.heat_adj_per[key_name]['y']
                dict_network[5]['x'] += self.heat_adj_org[key_name]['x']
                dict_network[5]['y'] += self.heat_adj_org[key_name]['y']
                dict_network[6]['x'] += self.heat_per_org[key_name]['x']
                dict_network[6]['y'] += self.heat_per_org[key_name]['y']

            folder_name = "{}/{}/{}/".format(DIR_OUTPUT, self.lfo.db_name, datetime.now().strftime('%Y-%m-%d'))
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
            date_file_name = "{}/{}-all".format(folder_name, datetime.now().strftime('%Y-%m-%d')) # %Y-%m-%d_%H-%M
            self.network_docs_report(dict_network, date_file_name, folder_name)

        else:
            for key_name in self.label_sent:
                dict_network = [{'label': 'per', 'x': self.heat_per[key_name]['x'], 'y': self.heat_per[key_name]['y']},
                               {'label': 'org', 'x': self.heat_org[key_name]['x'], 'y': self.heat_org[key_name]['y']},
                               {'label': 'per_verb', 'x': self.heat_verb_per[key_name]['x'],
                                'y': self.heat_verb_per[key_name]['y']},
                               {'label': 'org_verb', 'x': self.heat_verb_org[key_name]['x'],
                                'y': self.heat_verb_org[key_name]['y']},
                               {'label': 'per_adj', 'x': self.heat_adj_per[key_name]['x'],
                                'y': self.heat_adj_per[key_name]['y']},
                               {'label': 'org_adj', 'x': self.heat_adj_org[key_name]['x'],
                                'y': self.heat_adj_org[key_name]['y']},
                                {'label': 'per_org', 'x': self.heat_per_org[key_name]['x'], 'y': self.heat_per_org[key_name]['y']}]

                folder_name = "{}/{}/{}/".format(DIR_OUTPUT, self.lfo.db_name, datetime.now().strftime('%Y-%m-%d'))
                if not os.path.exists(folder_name):
                    os.makedirs(folder_name)
                date_file_name = "{}/{}-{}".format(folder_name, datetime.now().strftime('%Y-%m-%d_%H-%M'), key_name)
                self.network_docs_report(dict_network, date_file_name)

    def save(self):
        date_file_name = DIR_OUTPUT + datetime.now().strftime('%Y-%m-%d_%H-%M')
        with open('{}_model.pkl'.format(date_file_name), 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    def save_db(self):
        db_client = DataAccessMongo()
        for item in self.label_sent:
            row = {'file': self.label_file[item], 'chunk': self.label_chunk_sent[item],
                   'entities_person': self.label_per_sent[item], 'entities_org': self.label_org_sent[item],
                   'verb': self.label_verb_sent[item], 'adj': self.label_adj_sent[item], 'heat_per': self.heat_per[item],
                   'heat_org': self.heat_org[item], 'heat_verb_per': self.heat_verb_per[item],
                   'heat_verb_org': self.heat_verb_org[item], 'heat_adj_per': self.heat_adj_per[item],
                   'heat_adj_org': self.heat_adj_org[item], 'heat_per_org': self.heat_per_org[item]}
            db_client.save(row, self.lfo.db_name, self.lfo.db_coll)

    @staticmethod
    def load(path_dump):
        with open(path_dump, 'rb') as input_obj:
            obj_restore = pickle.load(input_obj)
            return obj_restore


if __name__ == '__main__':
    path = "/media/gabriel/Data/temp/scraping-master/sentencias_JyP/ext_data/output/sentencias/"
    ls = LoadSource(db_name='sentencias_jyp', db_coll='document', path=path, filter_files=[68])
    ls.load_files_jyp()
    ls.print_sentences_load()
    filter_pages_data = [305, 306]  # [item for item in range(305, 1810)]  # [305, 306]
    bs = BuildSource(load_files_object=ls, filter_pages=[{1: filter_pages_data}])
    bs.parser_document()
    bs.print_sentences_parser()
    bs.analyzer_doc()
    # bs.graph_docs()
    bs.compare_docs(all_docs=True)
    bs.network_docs(all_docs=True)



