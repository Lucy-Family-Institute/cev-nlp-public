import os
from preprocess import PreprocessPipeline
import spacy
import re
from interviews.core import Interview
import json
import pprint

path = os.path.abspath(os.path.dirname(__file__))

# Load the data:

path_interviews = "/home/vvargas/corpus/raw/04_entrevistas/all_interviews_2021_03_18_all_leves.json"
# path_interviews = "/home/vladimir/Documents/Projects/ComisionDeLaVerdad/survey-analysis/src/co-occurrence/import/input/survey_labelled_sample_500_level_4.json"
raw_interviews_path = os.path.join(path, "..", "data", "interviews_raw.json")
train_json_path = os.path.join(path, "..", "data", "interviews_preprocessed.json")
train_path = os.path.join(path, "..", "data", "interviews_train.txt")
ids = []
interventions = 0
if os.path.exists(raw_interviews_path):
    with open(raw_interviews_path, "r") as f:
        for line in f.readlines():
            line = json.loads(line)
            ids.append(line["identifier"])
with open(path_interviews, "r") as interviews:
    with open(raw_interviews_path, "a") as raw:
        for line in interviews.readlines():
            line = json.loads(line)
            _id = line["identifier"]
            if _id in ids:
                continue
            ids.append(_id)
            try:
                content = line["labelled"]["content"]
            except (TypeError, KeyError):
                pass
            try:
                content = line["content"]
            except (TypeError, KeyError):
                # TODO: save this info in a log file (JSON-like)
                print("********* NO CONTENT IN THIS LINE *************")
                pprint.pprint(line)
                print("********************************")
                continue
            interview = Interview(_id=_id, text=content)
            # TODO: modify get_dialogue to save errors in a log file (JSON-like)
            dialogue = interview.get_dialogue(verbose=True)
            interventions_list = []
            for intervention in dialogue:
                if "TEST" not in intervention["interlocutor"]:
                    continue
                text = intervention["text"]
                if len(text) < 50:
                    continue
                interventions_list.append(text.replace("\n", " "))
                interventions += 1
            if len(interventions_list) > 0:
                interview_json = {"identifier": _id, "witness_interventions": interventions_list}
                raw.write(json.dumps(interview_json))
                raw.write("\n")
del ids

print(f"A total of {interventions} lines were obtained from the interviews.")

# Clean the data:
try:
    assert isinstance(interventions, int)
except:
    interventions = 268298
try:
    nlp = spacy.load("es_dep_news_trf",  disable=["parser", "ner"])
    # nlp = spacy.load("es_core_news_sm",  disable=["parser", "ner"])
except OSError:
    raise OSError(
        "Please, before running this code, run `python -m spacy download es_dep_news_trf`"
    )
pp = PreprocessPipeline(tokenise=False, nlp=nlp, symbols=["¿", "¡"])

BUFSIZE = 1000000
processed = 0

# TODO Check if train_json_path exists, and do not preprocess those that will be processed by the same model
with open(raw_interviews_path, "r") as raw:
    with open(train_json_path, "w") as train:
        tmp_lines = raw.readlines(BUFSIZE)
        while tmp_lines:
            json_lines = [json.loads(line) for line in tmp_lines]
            result = pp([t for jline in json_lines for t in jline["witness_interventions"]], num_cpus=-1)
            start = 0
            end = 0
            for jline in json_lines:
                plines = len(jline["witness_interventions"])
                processed += plines
                end += plines
                jline["preprocessed_interventions"] = result[start: end]
                jline["model"] = "es_dep_news_trf"
                start = end
                train.write(json.dumps(jline))
                train.write("\n")

            tmp_lines = raw.readlines(BUFSIZE)
            print(
                f"{processed} lines processed. {interventions-processed} more lines to go."
            )

# Create training file for FastText

with open(train_json_path, 'r') as train:
    with open(train_path, 'w') as ft:
        for interview in train.readlines():
            interventions = json.loads(interview)["preprocessed_interventions"]
            for line in interventions:
                if len(line) > 50:   
                    line = re.sub('\s+', ' ', line).strip() 
                    ft.write(line)
                    if not line.endswith("\n"):
                        ft.write("\n")
