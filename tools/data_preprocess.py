import re
import hazm
import string
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import load_dataset, load_metric, Dataset, concatenate_datasets, load_from_disk, DatasetDict
from datasets import Audio
from num2words import num2words as words



_normalizer = hazm.Normalizer()


chars_to_ignore = [
    ",", ".", "!", "-", ";", ":", '""', "%", "'", '"', "�",
    "#", "!", "؟", "?", "«", "»", "،", "(", ")", "؛", "'ٔ", "٬",'ٔ', ",", "?",
    ".", "!", "-", ";", ":",'"',"“", "%", "‘", "”", "�", "–", "…", "_", "”", '“', '„',
    'ā', 'š',
]

chars_to_ignore = chars_to_ignore + list(string.ascii_lowercase + string.digits)

chars_to_mapping = {
    'ك': 'ک', 'دِ': 'د', 'بِ': 'ب', 'زِ': 'ز', 'ذِ': 'ذ', 'شِ': 'ش', 'سِ': 'س', 'ى': 'ی',
    'ي': 'ی', 'أ': 'ا', 'ؤ': 'و', "ے": "ی", "ۀ": "ه", "ﭘ": "پ", "ﮐ": "ک", "ﯽ": "ی",
    "ﺎ": "ا", "ﺑ": "ب", "ﺘ": "ت", "ﺧ": "خ", "ﺩ": "د", "ﺱ": "س", "ﻀ": "ض", "ﻌ": "ع",
    "ﻟ": "ل", "ﻡ": "م", "ﻢ": "م", "ﻪ": "ه", "ﻮ": "و", 'ﺍ': "ا", 'ة': "ه",
    'ﯾ': "ی", 'ﯿ': "ی", 'ﺒ': "ب", 'ﺖ': "ت", 'ﺪ': "د", 'ﺮ': "ر", 'ﺴ': "س", 'ﺷ': "ش",
    'ﺸ': "ش", 'ﻋ': "ع", 'ﻤ': "م", 'ﻥ': "ن", 'ﻧ': "ن", 'ﻭ': "و", 'ﺭ': "ر", "ﮔ": "گ",
    "۱۴ام": "۱۴ ام",

    "a": " ای ", "b": " بی ", "c": " سی ", "d": " دی ", "e": " ایی ", "f": " اف ",
    "g": " جی ", "h": " اچ ", "i": " آی ", "j": " جی ", "k": " کی ", "l": " ال ",
    "m": " ام ", "n": " ان ", "o": " او ", "p": " پی ", "q": " کیو ", "r": " آر ",
    "s": " اس ", "t": " تی ", "u": " یو ", "v": " وی ", "w": " دبلیو ", "x": " اکس ",
    "y": " وای ", "z": " زد ",
    "\u200c": " ", "\u200d": " ", "\u200e": " ", "\u200f": " ", "\ufeff": " ", "\u202b": " ", 
}


def map_words_dict(csv_file="map_words.csv"):
    map_df = pd.DataFrame(pd.read_csv("tools/map_words.csv"))
    mapping_dict = {}
    for i in range(len(map_df['original'])):
        mapping_dict[map_df['original'][i]] = map_df['corrected'][i]
    return mapping_dict

def multiple_replace_words(text, mapping_dict):
    pattern = "|".join(map(re.escape, mapping_dict.keys()))
    return re.sub(pattern, lambda m: mapping_dict[m.group()], str(text))

def multiple_replace(text, chars_to_mapping):
    pattern = "|".join(map(re.escape, chars_to_mapping.keys()))
    return re.sub(pattern, lambda m: chars_to_mapping[m.group()], str(text))

def remove_special_characters(text, chars_to_ignore_regex):
    text = re.sub(chars_to_ignore_regex, '', text).lower() + " "
    return text

words_to_mapping = map_words_dict()

def normalizer(row, chars_to_ignore=chars_to_ignore, chars_to_mapping=chars_to_mapping, words_to_mapping=words_to_mapping):
    text = row['sentence']
    chars_to_ignore_regex = f"""[{"".join(chars_to_ignore)}]"""
    text = text.lower().strip()

    text = _normalizer.normalize(text)
    text = multiple_replace_words(text, words_to_mapping)
    text = multiple_replace(text, chars_to_mapping)
    text = remove_special_characters(text, chars_to_ignore_regex)
    text = re.sub(" +", " ", text)
    _text = []
    for word in text.split():
        try:
            word = int(word)
            _text.append(words(word, lang='fa'))
            # print("num2word")
        except:
            _text.append(word)

    text = " ".join(map(str, _text)) + " "
    text = text.strip()

    if not len(text) > 0:
        return None

    row['sentence'] = text
    return row


def read_dataset():

    mrdataset = DatasetDict()

    # dataset = load_dataset()

    # dataset = load_dataset('tsv', data_files={'train': "/home/eri-3090/Documents/mrpeyghan/Peyghan/ASR/whisper/train_test_splitter/output/train_split.tsv", 'test': "/home/eri-3090/Documents/mrpeyghan/Peyghan/ASR/whisper/train_test_splitter/output/test_split.tsv"})


    data_csv = pd.read_csv("/home/eri-3090/Documents/mrpeyghan/Peyghan/ASR/whisper/train_test_splitter/output/train_split.tsv", sep="\t", on_bad_lines='skip')
    data_df = pd.DataFrame(data_csv)
    mrdataset["train"] = Dataset.from_pandas(data_df)

    data_csv = pd.read_csv("/home/eri-3090/Documents/mrpeyghan/Peyghan/ASR/whisper/train_test_splitter/output/test_split.tsv", sep="\t", on_bad_lines='skip')
    data_df = pd.DataFrame(data_csv)
    mrdataset["validation"] = Dataset.from_pandas(data_df)
    
    mrdataset = mrdataset.map(normalizer)
    mrdataset["train"].to_csv("/home/eri-3090/Documents/mrpeyghan/Peyghan/ASR/whisper/train_test_splitter/output/train_split_norm.tsv", sep="\t", index=False)
    mrdataset["validation"].to_csv("/home/eri-3090/Documents/mrpeyghan/Peyghan/ASR/whisper/train_test_splitter/output/test_split_norm.tsv", sep="\t", index=False)
    # mrdataset["train"] = mrdataset["train"].remove_columns(["accents", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes", "variant"])
    # mrdataset["validation"] = mrdataset["validation"].remove_columns(['client_id', 'variant'])

    # print(mrdataset)

    mrdataset = mrdataset.cast_column("path", Audio(sampling_rate=16000))

    return mrdataset

if __name__ == "__main__":
    mrdataset = read_dataset()
    print(mrdataset)

