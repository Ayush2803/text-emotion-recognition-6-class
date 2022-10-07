import torch
from torch.utils.data import Dataset
from transformers import RobertaTokenizer, RobertaModel
import pandas as pd
import json
import os
from tqdm import tqdm
import pickle
import logging
import argparse

def get_emotion2id(num_classes, DATASET="IEMOCAP"):
    """Get a dict that converts string class to numbers."""
    if DATASET == "IEMOCAP":
        if num_classes==4:
          emotion2id= {
              "neutral": 0,
              "frustration": 1,
              "sadness": 2,
              "anger":1,
              "excited":3,
              "happiness":3
          }
        if num_classes==5:
          emotion2id= {
              "neutral": 0,
              "frustration": 1,
              "sadness": 2,
              "anger":3,
              "excited":4,
              "happiness":4
          }
        if num_classes==6:
          emotion2id= {
              "neutral": 0,
              "frustration": 1,
              "sadness": 2,
              "anger":3,
              "excited":4,
              "happiness":5
          }
        id2emotion = {val: key for key, val in emotion2id.items()}

    return emotion2id, id2emotion


class extract_dialogues:

  def __init__(self,root_dir, split, speaker_mode='UPPER', num_classes=6, past=1, future=0, dataset="IEMOCAP"):
    self.ROOT_DIR = root_dir
    self.DATASET= dataset
    self.SPLIT= split
    self.inputs_ = []
    self.speaker_mode= speaker_mode
    self.num_past_utterances = past
    self.num_future_utterances = future
    self.emotion2id, self.id2emotion = get_emotion2id(num_classes,self.DATASET)
    self._load_emotions()
    self._load_utterance_ordered()
    self._string2tokens()
    data_path= root_dir+"/data-"+str(num_classes)+'-'+str(past) + "/"
    os.makedirs(data_path, exist_ok=True)
    fname = data_path+split+ '.pkl'
    with open(fname, 'wb') as writefile:
      pickle.dump(self.inputs_,writefile) 

  def _load_emotions(self):
      """Load the supervised labels"""
      if self.DATASET in ["MELD", "IEMOCAP"]:
          with open(
              os.path.join(self.ROOT_DIR, self.DATASET, "emotions.json"), "r"
          ) as stream:
              self.emotions = json.load(stream)[self.SPLIT]


  def _load_utterance_ordered(self):
      """Load the ids of the utterances in order."""
      if self.DATASET in ["MELD", "IEMOCAP"]:
          path = os.path.join(self.ROOT_DIR, self.DATASET, "utterance-ordered.json")
      elif self.DATASET == "MELD_IEMOCAP":
          path = "./utterance-ordered-MELD_IEMOCAP.json"

      with open(path, "r") as stream:
          self.utterance_ordered = json.load(stream)[self.SPLIT]


  def _load_utterance_speaker_emotion(self, uttid, speaker_mode) -> dict:
      """Load an speaker-name prepended utterance and emotion label"""

      if self.DATASET in ["MELD", "IEMOCAP"]:
          text_path = os.path.join(
              self.ROOT_DIR, self.DATASET, "raw-texts", self.SPLIT, uttid + ".json"
          )
      elif self.DATASET == "MELD_IEMOCAP":
          assert len(uttid.split("/")) == 4
          d_, s_, d__, u_ = uttid.split("/")
          text_path = os.path.join(self.ROOT_DIR, d_, "raw-texts", s_, u_ + ".json")

      with open(text_path, "r") as stream:
          text = json.load(stream)

      utterance = text["Utterance"].strip()
      emotion = text["Emotion"]

      if self.DATASET == "MELD":
          speaker = text["Speaker"]
      elif self.DATASET == "IEMOCAP":
          sessid = text["SessionID"]
          # https: // www.ssa.gov/oact/babynames/decades/century.html
          speaker = {
              "Ses01": {"Female": "Mary", "Male": "James"},
              "Ses02": {"Female": "Patricia", "Male": "John"},
              "Ses03": {"Female": "Jennifer", "Male": "Robert"},
              "Ses04": {"Female": "Linda", "Male": "Michael"},
              "Ses05": {"Female": "Elizabeth", "Male": "William"},
          }[sessid][text["Speaker"]]
      elif self.DATASET == "MELD_IEMOCAP":
          speaker = ""
      else:
          raise ValueError(f"{self.DATASET} not supported!!!!!!")

      if speaker_mode is not None and speaker_mode.lower() == "upper":
          utterance = speaker.upper() + ": " + utterance
      elif speaker_mode is not None and speaker_mode.lower() == "title":
          utterance = speaker.title() + ": " + utterance

      return {"Utterance": utterance, "Emotion": emotion}

  def _create_input(
      self, diaids, speaker_mode, num_past_utterances, num_future_utterances
  ):
      """Create an input which will be an input to RoBERTa."""

      args = {
          "diaids": diaids,
          "speaker_mode": speaker_mode,
          "num_past_utterances": num_past_utterances,
          "num_future_utterances": num_future_utterances,
      }
      tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
      max_model_input_size = tokenizer.max_model_input_sizes["roberta-large"]
      num_truncated = 0

      inputs = []
      for diaid in tqdm(diaids):
          ues = [
              self._load_utterance_speaker_emotion(uttid, speaker_mode)
              for uttid in self.utterance_ordered[diaid]
          ]

          num_tokens = [len(tokenizer(ue["Utterance"])["input_ids"]) for ue in ues]

          for idx, ue in enumerate(ues):
              if ue["Emotion"] not in list(self.emotion2id.keys()):
                  continue

              label = self.emotion2id[ue["Emotion"]]

              indexes = [idx]
              indexes_past = [
                  i for i in range(idx - 1, idx - num_past_utterances - 1, -1)
              ]
              indexes_future = [
                  i for i in range(idx + 1, idx + num_future_utterances + 1, 1)
              ]

              offset = 0
              if len(indexes_past) < len(indexes_future):
                  for _ in range(len(indexes_future) - len(indexes_past)):
                      indexes_past.append(None)
              elif len(indexes_past) > len(indexes_future):
                  for _ in range(len(indexes_past) - len(indexes_future)):
                      indexes_future.append(None)

              for i, j in zip(indexes_past, indexes_future):
                  if i is not None and i >= 0:
                      indexes.insert(0, i)
                      offset += 1
                      if (
                          sum([num_tokens[idx_] for idx_ in indexes])
                          > max_model_input_size
                      ):
                          del indexes[0]
                          offset -= 1
                          num_truncated += 1
                          break
                  if j is not None and j < len(ues):
                      indexes.append(j)
                      if (
                          sum([num_tokens[idx_] for idx_ in indexes])
                          > max_model_input_size
                      ):
                          del indexes[-1]
                          num_truncated += 1
                          break

              utterances = [ues[idx_]["Utterance"] for idx_ in indexes]

              if num_past_utterances == 0 and num_future_utterances == 0:
                  assert len(utterances) == 1
                  final_utterance = utterances[0]

              elif num_past_utterances > 0 and num_future_utterances == 0:
                  if len(utterances) == 1:
                      final_utterance = "</s></s>" + utterances[-1]
                  else:
                      final_utterance = (
                          " ".join(utterances[:-1]) + "</s></s>" + utterances[-1]
                      )

              elif num_past_utterances == 0 and num_future_utterances > 0:
                  if len(utterances) == 1:
                      final_utterance = utterances[0] + "</s></s>"
                  else:
                      final_utterance = (
                          utterances[0] + "</s></s>" + " ".join(utterances[1:])
                      )

              elif num_past_utterances > 0 and num_future_utterances > 0:
                  if len(utterances) == 1:
                      final_utterance = "</s></s>" + utterances[0] + "</s></s>"
                  else:
                      final_utterance = (
                          " ".join(utterances[:offset])
                          + "</s></s>"
                          + utterances[offset]
                          + "</s></s>"
                          + " ".join(utterances[offset + 1 :])
                      )
              else:
                  raise ValueError

              input_ids_attention_mask = tokenizer(final_utterance)
              input_ids = input_ids_attention_mask["input_ids"]
              attention_mask = input_ids_attention_mask["attention_mask"]

              input_ = {
                  'text': final_utterance,
                  'label': label
              }

              inputs.append(input_)

      return inputs

  def _string2tokens(self):
      """Convert string to (BPE) tokens."""
      diaids = sorted(list(self.utterance_ordered.keys()))
      self.inputs_ = self._create_input(
          diaids=diaids,
          speaker_mode=self.speaker_mode,
          num_past_utterances=self.num_past_utterances,
          num_future_utterances=self.num_future_utterances,
      )

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--num_classes', type=int, default=6,help='num__of_classes')
  parser.add_argument('--past', type=int, default=2,help='past_utterances')
  parser.add_argument('--root_dir', type=str, default='data/',help='path of data')
  return parser.parse_args()

if __name__ == "__main__":
  args = parse_args()
  args.seeds = [0]
  print(args)

  train_extractor = extract_dialogues(root_dir=args.root_dir, split='train', num_classes=args.num_classes, past=args.past)
  val_extractor = extract_dialogues(root_dir=args.root_dir, split='val', num_classes=args.num_classes, past=args.past)
  test_extractor = extract_dialogues(root_dir=args.root_dir, split='test', num_classes=args.num_classes, past=args.past)