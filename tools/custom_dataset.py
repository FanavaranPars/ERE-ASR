import pandas as pd
import librosa 
from torch.utils.data import Dataset
from tools.data_preprocess import normalizer
from transformers import WhisperFeatureExtractor, WhisperTokenizer
from tqdm.auto import tqdm
import os


feature_extractor = WhisperFeatureExtractor.from_pretrained("whisper-large-v3")
tokenizer = WhisperTokenizer.from_pretrained("whisper-large-v3", language="fa", task="transcribe")


class custom_asr_dataset(Dataset):

    def __init__(self, data_annotation):

        self.annot = pd.DataFrame(pd.read_csv(data_annotation, sep="\t"))
    
    def __len__(self):
        return len(self.annot['path'])

    def __getitem__(self, idx):
        head_path, _ = os.path.split(os.getcwd())
        audio_path = os.path.join(head_path, 'dataset/clips', self.annot['path'][idx])
        audio, _ = librosa.load(audio_path, sr=16000, mono=True) 
        in_features = feature_extractor(audio, sampling_rate=16000).input_features[0]

        sentence = self.annot['sentence'][idx]
        labels = tokenizer(sentence).input_ids

        feature = {"input_features": in_features, "labels": labels}

        return feature




if __name__ == "__main__":
    annot_address = "train_test_splitter/output/cv/test_split_4090.tsv"
    len_annot = pd.read_csv(annot_address, sep="\t", on_bad_lines='skip')

    asr_dataset = custom_asr_dataset(data_annotation=annot_address)
    len_label = []
    for i in tqdm(range(asr_dataset.__len__())):
        feat = asr_dataset[i]
        len_label.append(len(feat["labels"]))
    print(max(len_label))


