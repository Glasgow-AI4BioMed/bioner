from bioc import biocxml
import gzip
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import DataCollatorForTokenClassification
from transformers import Trainer, TrainingArguments, TrainerCallback, EarlyStoppingCallback
from transformers import AutoConfig
import numpy as np
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score
from datasets import Dataset
import os
import argparse
import socket
import shutil

from model_preparation import prepare_model_repo


def tokenize_and_label(text, spans, tokenizer, label2id, word_based):

    # Tokenize with offset mapping only
    encoded = tokenizer(
        text,
        return_offsets_mapping=True,
        return_tensors=None,
        truncation=True,
    )
    
    offset_mapping = encoded["offset_mapping"]
    
    word_ids = encoded.word_ids()
    word_start_idxs = set( i for i,word_id in enumerate(word_ids) if word_id is not None and word_ids[i - 1] != word_id )
    
    if word_based:
        labels = [-100] * len(offset_mapping)
        for idx in word_start_idxs:
            labels[idx] = label2id['O']
    else:
        labels = [ -100 if start==end else label2id['O'] for start,end in offset_mapping ]

    for start, end, label in spans:
        token_idxs = [ i for i,(t_start,t_end) in enumerate(offset_mapping) if start <= t_start and t_end <= end ]

        if token_idxs: # If we've got tokens (otherwise the text has likely been truncated before getting to this span)
            labels[token_idxs[0]] = label2id[f'B-{label}']
            for token_idx in token_idxs[1:]:
                if token_idx in word_start_idxs or (not word_based):
                    labels[token_idx] = label2id[f'I-{label}']

    encoded["labels"] = labels
    encoded.pop("offset_mapping")
    return encoded


def make_dataset(collection, tokenizer, label2id, word_based):
    dataset = []
    
    for doc in collection.documents:
        for passage in doc.passages:
            
            spans = []
            for anno in passage.annotations:
                loc = anno.total_span
                start,end = loc.offset-passage.offset, loc.offset+loc.length-passage.offset
                spans.append( (start,end,anno.infons['label']) )
    
            text = passage.text
    
            tokenized = tokenize_and_label(text, spans, tokenizer, label2id, word_based)
            dataset.append(tokenized)

    return dataset


def compute_metrics(eval_pred):
    """
    Compute metrics for NER evaluation using seqeval.
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)

    labels = labels.reshape(-1)
    predictions = predictions.reshape(-1)
    
    #print(f"{labels.shape=} {predictions.shape=}")
    #assert False
    
    # Remove ignored index (special tokens) and convert to labels
    nonmasked_predictions = [
        prediction for prediction, label in zip(predictions, labels) if label != -100
    ]
    
    nonmasked_labels = [
        label for prediction, label in zip(predictions, labels) if label != -100
    ]
    
    results = {
        "macro_precision": precision_score(nonmasked_labels, nonmasked_predictions, average='macro', zero_division=0.0),
        "macro_recall": recall_score(nonmasked_labels, nonmasked_predictions, average='macro', zero_division=0.0),
        "macro_f1": f1_score(nonmasked_labels, nonmasked_predictions, average='macro', zero_division=0.0),
        "accuracy": accuracy_score(nonmasked_labels, nonmasked_predictions),
    }
    return results



class SaveBestModelCallback(TrainerCallback):
    def __init__(self, save_path="./best_model"):
        self.save_path = save_path
        self.best_metric = None
        self.trainer = None
        
    def set_trainer(self, trainer):
        self.trainer = trainer
        
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        assert metrics

        # Get the metric name and comparison direction from TrainingArguments
        metric_name = self.trainer.args.metric_for_best_model
        assert metric_name, "Warning: 'metric_for_best_model' not set in TrainingArguments."

        current_metric = metrics.get(metric_name)
        assert current_metric, f"Warning: Metric '{metric_name}' not found in evaluation metrics."

        greater_is_better = self.trainer.args.greater_is_better

        is_better = (
            self.best_metric is None or
            (greater_is_better and current_metric > self.best_metric) or
            (not greater_is_better and current_metric < self.best_metric)
        )

        if is_better:
            self.best_metric = current_metric
            print(f"\n\nNew best model found! {metric_name} = {current_metric:.4f} at epoch {state.epoch:.0f}. Saving to {self.save_path}\n")
            self.trainer.save_model(self.save_path)
            with open(f'{self.save_path}/epoch.txt','w') as f:
                f.write(f"{state.epoch:.0f}")


def run_classification_report(trainer, dataset, id2label, labels):

    results = trainer.predict(dataset)

    label_ids = results.label_ids.reshape(-1)
    predictions = np.argmax(results.predictions, axis=2).reshape(-1)

    nonmasked_predictions = [
        prediction for prediction, label_id in zip(predictions, label_ids) if label_id != -100
    ]
    
    nonmasked_label_ids = [
        label_id for prediction, label_id in zip(predictions, label_ids) if label_id != -100
    ]
    
    report_dict = classification_report(nonmasked_label_ids, nonmasked_predictions, labels=sorted(id2label.keys()), target_names=labels, zero_division=0.0, output_dict=True)

    return report_dict

def train_and_tune_model(base_model, model_name, annotated_labels, train_collection, val_collection, test_collection, n_trials, wandb_name, word_based):
    
    labels = ['O'] + [ f'{prefix}-{label}' for label in annotated_labels for prefix in ['B','I'] ]
    id2label = { idx:label for idx,label in enumerate(labels) }
    label2id = { label:idx for idx,label in enumerate(labels) }

    print(f"{id2label=}")

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    # Deal with the model not have the max_length saved (which gives a warning)
    model_config = AutoConfig.from_pretrained(base_model)
    tokenizer.model_max_length = model_config.max_position_embeddings

    train_tokenized = make_dataset(train_collection, tokenizer, label2id, word_based)
    val_tokenized = make_dataset(val_collection, tokenizer, label2id, word_based)
    test_tokenized = make_dataset(test_collection, tokenizer, label2id, word_based)

    train_dataset = Dataset.from_list(train_tokenized)
    val_dataset = Dataset.from_list(val_tokenized)
    test_dataset = Dataset.from_list(test_tokenized)

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    if wandb_name:
        os.environ["WANDB_PROJECT"] = wandb_name

    tokenizer.save_pretrained(model_name)
    
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=3,
        early_stopping_threshold=0.001
    )
    
    save_best_model_callback = SaveBestModelCallback(model_name)
    
    unique_info = f'{socket.gethostname()}_{os.getpid()}'
    tmp_model_dir = f"tmp_mentiondetector_{unique_info}"
    training_args = TrainingArguments(
        output_dir=tmp_model_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        metric_for_best_model="eval_macro_f1",
        load_best_model_at_end=True,
        greater_is_better=True,
        seed=42,
        num_train_epochs=1,
        report_to=("wandb" if wandb_name else "none")
    )

    def model_init():
        return AutoModelForTokenClassification.from_pretrained(
            base_model, id2label=id2label
        )
    
    def hp_space(trial):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
            "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8, 16]),
            "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.3),
            "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.2),
        }
    
    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping_callback, save_best_model_callback],
    )
    
    save_best_model_callback.set_trainer(trainer)

    best_trial = trainer.hyperparameter_search(
        hp_space=hp_space,
        backend="optuna",
        compute_objective=lambda metrics: metrics["eval_macro_f1"],
        n_trials=n_trials,
        direction="maximize",
    )

    # Remove temporary directory
    shutil.rmtree(tmp_model_dir)

    # Load up the best model
    trainer.model = AutoModelForTokenClassification.from_pretrained(model_name).to(trainer.args.device)
    
    train_token_report = run_classification_report(trainer, train_dataset, id2label, labels)
    val_token_report = run_classification_report(trainer, val_dataset, id2label, labels)
    test_token_report = run_classification_report(trainer, test_dataset, id2label, labels)

    return best_trial.hyperparameters, train_token_report, val_token_report, test_token_report
                 
def main():
    parser = argparse.ArgumentParser('Run hyperparameter tuning for an NER model and save out the best')
    parser.add_argument('--train_corpus',type=str,required=True,help='Gzipped BioC XML corpus for training')
    parser.add_argument('--val_corpus',type=str,required=True,help='Gzipped BioC XML corpus for validation')
    parser.add_argument('--test_corpus',type=str,required=True,help='Gzipped BioC XML corpus for testing')
    parser.add_argument('--base_model',type=str,required=False,default='microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract',help='Base BERT model to train from')
    parser.add_argument('--n_trials',type=int,required=True,help='Number of trials to run when tuning')
    parser.add_argument('--wandb_name',type=str,required=False,help="Project name for wandb (or don't use wandb if not provided)")
    parser.add_argument('--model_name',type=str,required=True,help='Name of model to save (and output directory)')
    parser.add_argument('--model_card_template',type=str,required=True,help='Markdown file with template of model_card')
    parser.add_argument('--dataset_info',type=str,required=True,help='Markdown file with dataset information')
    parser.add_argument('--word_based',action='store_true',required=False,help='Whether to train a word-based model (only labels first token of each word)')
    args = parser.parse_args()

    with gzip.open(args.train_corpus, 'rt', encoding='utf8') as f:
        train_collection = biocxml.load(f)
    with gzip.open(args.val_corpus, 'rt', encoding='utf8') as f:
        val_collection = biocxml.load(f)
    with gzip.open(args.test_corpus, 'rt', encoding='utf8') as f:
        test_collection = biocxml.load(f)

    
    annotated_labels = sorted(set( anno.infons['label'] for doc in train_collection.documents+val_collection.documents+test_collection.documents for passage in doc.passages for anno in passage.annotations ))

    best_hyperparameters, train_token_report, val_token_report, test_token_report = train_and_tune_model(args.base_model, args.model_name, annotated_labels, train_collection, val_collection, test_collection, args.n_trials, args.wandb_name, bool(args.word_based))
    
    prepare_model_repo(args.model_name, args.base_model, annotated_labels, args.n_trials, best_hyperparameters, train_collection, val_collection, test_collection, train_token_report, val_token_report, test_token_report, args.model_card_template, args.dataset_info, bool(args.word_based))

    print("Done.")
    
    
if __name__ == '__main__':
    main()

