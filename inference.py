from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa 
import pandas as pd
from tqdm.auto import tqdm
import os
import argparse

# Create a directory to store inference results if it doesn't exist
os.makedirs('inference_results', exist_ok=True)

# Set up argument parser
parser = argparse.ArgumentParser(prog='persian whisper',
                                 description='inference for persian whisper')

# Add arguments to the parser
parser.add_argument('--device', default="cuda", type=str, choices=['cuda', 'cpu'], help='whether to run the code on cpu or gpu(cuda)')
parser.add_argument('--ckp', default="checkpoint/fa-large-v3", type=str, help='path to the model checkpoint using for inference')
parser.add_argument('--mode', default='sample', type=str, choices=['sample', 'csv'], help='choose to inference a single sample or a csv file')
parser.add_argument('--path_audio', default=None, type=str, help='path to your audio file')
parser.add_argument('--path_csv', default=None, type=str, help="path to your csv file (csv must include 'path' and 'sentence' columns)")
parser.add_argument('--save_path', default='inference_results', type=str, help='path to where you want to save transcriptions')

# Parse the arguments
args = parser.parse_args()

class fawhisper_inference():
    def __init__(self, model_name: str="whisper-large-v3", checkpoint: str=args.ckp):
        # Initialize the Whisper processor with the specified model and language
        self.processor = WhisperProcessor.from_pretrained(model_name, language="fa", task="transcribe")
        self.model_checkpoint = checkpoint
        # Load the Whisper model from the specified checkpoint and move it to the specified device (CPU/GPU)
        self.model = WhisperForConditionalGeneration.from_pretrained(self.model_checkpoint, use_cache = False).to(args.device)

    def inference_sample(self, path: str):
        # Load the audio file from the specified path
        waveform, sample_rate = librosa.load(path, sr=16000)
        # Process the audio to get input features for the model
        input_features = self.processor(waveform, sampling_rate=sample_rate, return_tensors="pt").input_features
        input_features = input_features.to(args.device)

        # Generate predicted transcription IDs
        predicted_ids = self.model.generate(input_features)

        # Decode the transcription IDs to get the transcription text
        transcription = self.processor.batch_decode(predicted_ids.detach().cpu(), skip_special_tokens=True)

        # Save the transcription to a file
        with open(os.path.join(args.save_path, "sample_transcription.txt"), "w") as f:
            f.write(transcription[0])

        return transcription[0]
    
    def inference_csv(self, csv_file: str, save_path: str=os.path.join(args.save_path, 'csv_transcriptions.tsv')):
        # Load the CSV file containing paths and sentences
        test_data = pd.DataFrame(pd.read_csv(csv_file, sep="\t"))

        results = {"sample": [], "label": [], "predicted": []}
        # Iterate over each row in the CSV file
        for i in tqdm(range(len(test_data['path']))):
            # Construct the full path to the audio file
            head_path, _ = os.path.split(os.getcwd())
            sample_data = os.path.join(head_path, 'dataset/clips', test_data['path'][i])
            results["sample"].append(sample_data)
            results["label"].append(test_data['sentence'][i])

            # Load the audio file
            waveform, sample_rate = librosa.load(sample_data, sr=16000)

            # Process the audio to get input features for the model
            input_features = self.processor(waveform, sampling_rate=sample_rate, return_tensors="pt").input_features
            input_features = input_features.to(args.device)
            
            # Generate predicted transcription IDs
            predicted_ids = self.model.generate(input_features)

            # Decode the transcription IDs to get the transcription text
            transcription = self.processor.batch_decode(predicted_ids.detach().cpu(), skip_special_tokens=True)

            # Store the transcription results
            results["predicted"].append(transcription[0])
        
            # Save the results to a TSV file
            results_df = pd.DataFrame(results)
            results_df.to_csv(save_path, index=False)

def main():    
    # Initialize the inference class
    fa_whisper = fawhisper_inference()

    # Perform inference based on the selected mode (sample or CSV)
    if args.mode == 'sample':
        sample_data = args.path_audio
        tanscript = fa_whisper.inference_sample(path=sample_data)
        print(tanscript)
    
    if args.mode == 'csv':
        fa_whisper.inference_csv(csv_file=args.path_csv)

# Execute the main function if the script is run directly
if __name__ == "__main__":
    main()

