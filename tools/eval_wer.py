from datasets import load_dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa 
import pandas as pd
from tqdm.auto import tqdm
from jiwer import wer
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
import pandas as pd
from custom_dataset import custom_asr_dataset
from tqdm.auto import tqdm


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
    map_df = pd.DataFrame(pd.read_csv("map_words.csv"))
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


# normalization
data_file = "/home/eri-3090/Documents/mrpeyghan/Peyghan/ASR/alignment/kaggleset/kaggle_dataset.csv"
mrdataset = DatasetDict()
data_csv = pd.read_csv(data_file, sep=",", on_bad_lines='skip')
# data_csv = pd.read_csv(data_file)
data_df = pd.DataFrame(data_csv)
# remove empty rows
print('dropping faulty rows ....')
for idx, data_sentence in enumerate(data_df['sentence']):
    if data_sentence == " " or len(data_sentence) <= 4:
        data_df = data_df.drop(idx)
        # print('empty row dropped')
print('Done')
mrdataset["test"] = Dataset.from_pandas(data_df)

mrdataset = mrdataset.map(normalizer, num_proc=12)
mrdataset["test"].to_csv(data_file.split('.')[0] + "_norm." + data_file.split('.')[1], sep=",", index=False)
# length fix
asr_dataset = custom_asr_dataset(data_file.split('.')[0] + "_norm." + data_file.split('.')[1])
df = pd.read_csv(data_file.split('.')[0] + "_norm." + data_file.split('.')[1], sep=",", on_bad_lines='skip')
for i in tqdm(range(len(df['path']))):
    feat = asr_dataset[i]
    if len(feat['labels']) > 128:
        print(f"len: {len(feat['labels'])}")
        print(f"removed sentence: {df['sentence'][i]}")
        df = df.drop(i)

df.to_csv(data_file.split('.')[0] + "_norm_fix." + data_file.split('.')[1], sep=",", index=False)

test_file = data_file.split('.')[0] + "_norm_fix." + data_file.split('.')[1]

# test_file = "/home/eri-3090/Documents/mrpeyghan/Peyghan/ASR/alignment/kaggleset/kaggle_dataset_norm_fix.csv"
def eval_model(model_checkpoint: str, test_file: str=test_file):
    test_data = pd.DataFrame(pd.read_csv(test_file, sep=","))

    processor = WhisperProcessor.from_pretrained("whisper-large-v3", language="fa", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(model_checkpoint, use_cache = False).to('cuda')

    sum_wer = 0
    faulty_inputs = []
    for i in tqdm(range(len(test_data['path']))):

        sample_data = test_data['path'][i]
        # results["sample"].append(sample_data)
        # results["label"].append(test_data['sentence'][i])

        waveform, sample_rate = librosa.load(sample_data, sr=16000)

        input_features = processor(
            waveform, sampling_rate=sample_rate, return_tensors="pt"
        ).input_features
        input_features = input_features.to("cuda")
        
        predicted_ids = model.generate(input_features)

        transcription = processor.batch_decode(predicted_ids.detach().cpu(), skip_special_tokens=True)

        # results["predicted"].append(transcription[0])

        sample_wer = wer(test_data['sentence'][i], transcription[0])
        if sample_wer >= 0.3:
            faulty_inputs.append(sample_data)
            print("Faulty Input Detected!")

        if sample_wer >= 1:
            sample_wer = 1
            print("anomaly Detected!")
            print(f"\n\n label: {test_data['sentence'][i]}\n predicted: {transcription[0]}")


        sum_wer += sample_wer

        if i % 50 == 0:
            print(f"avg WER: {sum_wer / (i + 1)}")

    avg_wer = sum_wer / len(test_data['path'])
    print(f"\n >>>   Average WER: {avg_wer}   <<< \n")

    return avg_wer, faulty_inputs

def main():
    # csv_file = "/home/eri-3090/Documents/mrpeyghan/Peyghan/ASR/testsets/7486182/AghrabeSet_norm_fix.tsv"
    model_checkpoint = "/home/eri-3090/Documents/mrpeyghan/Peyghan/ASR/whisper/trained_versions/whisper_large_v12_23May21_fa/checkpoint-42000"

    avg_wer, faulty_inputs = eval_model(model_checkpoint=model_checkpoint)
    print(f"\n >>>   Average WER: {avg_wer}   <<< \n")

if __name__ == "__main__":
    main()


    
