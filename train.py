from tools.utils import DataCollatorSpeechSeq2SeqWithPadding, compute_metrics
from transformers import WhisperProcessor
from transformers import WhisperForConditionalGeneration
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer
from tools.custom_dataset import custom_asr_dataset
import argparse
from datetime import datetime

# Set up argument parser
parser = argparse.ArgumentParser(prog='whisper_farsi', 
                                 description='fine-tuning of multilingual whisper (large-v3) on persian language')

# Add arguments to the parser
parser.add_argument('-v', '--version', default=datetime.today().strftime('DATE%Y-%m-%dTIME%H:%M:%S'), type=str, help='define a version for the training (default: Date Time)')
parser.add_argument('--bs', default=16, type=int, help='training batch size')
parser.add_argument('--lr', default=1e-05, type=float, help='learning rate')
parser.add_argument('--maxsteps', default=60000, type=int, help='maximum number of training steps')
parser.add_argument('--evalsteps', default=6000, type=int, help='evaluate model each evalsteps training step')
parser.add_argument('--savesteps', default=6000, type=int, help='save model each evalsteps training step')
parser.add_argument('--train_csv', default="data_annot/train_split.tsv", type=str, help='path to your train data annotation')
parser.add_argument('--test_csv', default="data_annot/test_split.tsv", type=str, help='path to your test data annotation')

# Parse the arguments
args = parser.parse_args()

# Define the save path for trained models
save_path = "trained_versions/"
# Set up training arguments for the Seq2Seq model
training_args = Seq2SeqTrainingArguments(
    output_dir= save_path + args.version,  # Output directory for the model
    per_device_train_batch_size=args.bs,  # Batch size for training
    gradient_accumulation_steps=2,  # Number of steps to accumulate gradients
    learning_rate=args.lr,  # Learning rate
    warmup_ratio=0.1,  # Ratio of steps to warm up the learning rate
    max_steps=args.maxsteps,  # Maximum number of training steps
    gradient_checkpointing=True,  # Enable gradient checkpointing
    fp16=True,  # Use mixed precision training
    evaluation_strategy="steps",  # Evaluate the model every few steps
    per_device_eval_batch_size=8,  # Batch size for evaluation
    predict_with_generate=True,  # Use generation for evaluation
    generation_max_length=128,  # Maximum length of generated sequences
    eval_steps=args.evalsteps,  # Evaluation steps
    save_steps=args.savesteps,  # Save steps
    logging_steps=100,  # Logging steps
    report_to=["tensorboard"],  # Report to TensorBoard
    metric_for_best_model="wer",  # Metric to use for selecting the best model
    greater_is_better=False,  # Lower is better for the metric
    optim='adafactor',  # Optimizer to use
    dataloader_num_workers=16,  # Number of workers for the data loader
)

def main(training_args=training_args):
    # Load the Whisper processor with Persian language settings
    processor = WhisperProcessor.from_pretrained("whisper-large-v3", language="fa", task="transcribe")

    # Load the training dataset
    train_path = args.train_csv
    train_dataset = custom_asr_dataset(data_annotation=train_path)

    # Load the validation dataset
    valid_path = args.test_csv
    valid_dataset = custom_asr_dataset(data_annotation=valid_path)

    # Create a data collator for padding sequences
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # Load the Whisper model
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3", use_cache = False)

    # Configure the model
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.generation_config.language = "fa"

    # Freeze the encoder to avoid training it
    model.freeze_encoder=True

    # Uncomment the following lines to apply augmentation
    # model.config.apply_spec_augment = True
    # model.config.mask_time_prob = 0.05
    # model.config.mask_feature_prob = 0.05

    # Set up the trainer with the specified arguments and datasets
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )
    
    # Save the training arguments to a JSON file
    with open(f'{save_path + args.version}/args.json', 'w') as fout:
        fout.write(training_args.to_json_string())

    # Start training the model
    trainer.train()

# Execute the main function if the script is run directly
if __name__ == "__main__":
    main()

