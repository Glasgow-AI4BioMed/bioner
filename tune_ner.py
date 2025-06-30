from bioc import biocxml
import gzip
from intervaltree import IntervalTree
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import DataCollatorForTokenClassification
from transformers import Trainer, TrainingArguments, TrainerCallback, EarlyStoppingCallback
import numpy as np
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score
from datasets import Dataset
import os
import json
import argparse

def tokenize_and_label(text, spans, tokenizer, label2id):

    tree = IntervalTree()
    for start,end,label in spans:
        tree.addi(start,end,label)

    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=True,
        padding=True,
        max_length=512
    )
                 
    label_ids = []
    
    prev_anno = None
    for start,end in encoding['offset_mapping']:
        cur_anno = tree[start:end]
        if start==end: # Special tokens that have zero length
            label_id = -100
        elif cur_anno:
            label = list(cur_anno)[0].data
                
            if cur_anno == prev_anno:
                label_id = label2id[f'I-{label}']
            else:
                label_id = label2id[f'B-{label}']
        else:
            label_id = label2id['O']
    
        label_ids.append(label_id)
        prev_anno = cur_anno

    encoding['labels'] = label_ids
    return encoding


def make_dataset(collection, tokenizer, label2id):
    dataset = []
    
    for doc in collection.documents:
        for passage in doc.passages:
            
            spans = []
            for anno in passage.annotations:
                loc = anno.total_span
                start,end = loc.offset-passage.offset, loc.offset+loc.length-passage.offset
                spans.append( (start,end,anno.infons['label']) )
    
            text = passage.text
    
            tokenized = tokenize_and_label(text, spans, tokenizer, label2id)
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
            print(f"New best model found! {metric_name} = {current_metric:.4f}. Saving to {self.save_path}")
            self.trainer.save_model(self.save_path)

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
    
    return classification_report(nonmasked_label_ids, nonmasked_predictions, labels=sorted(id2label.keys()), target_names=labels, zero_division=0.0)
    

def main():
    parser = argparse.ArgumentParser('Run hyperparameter tuning for an NER model and save out the best')
    parser.add_argument('--train_corpus',type=str,required=True,help='Gzipped BioC XML corpus for training')
    parser.add_argument('--val_corpus',type=str,required=True,help='Gzipped BioC XML corpus for validation')
    parser.add_argument('--test_corpus',type=str,required=True,help='Gzipped BioC XML corpus for testing')
    parser.add_argument('--n_trials',type=int,required=True,help='Number of trials to run when tuning')
    parser.add_argument('--wandb_name',type=str,required=False,help="Project name for wandb (or don't use wandb if not provided)")
    parser.add_argument('--output_dir',type=str,required=True,help='Directory to save final model to')
    args = parser.parse_args()

    with gzip.open(args.train_corpus, 'rt', encoding='utf8') as f:
        train_collection = biocxml.load(f)
    with gzip.open(args.val_corpus, 'rt', encoding='utf8') as f:
        val_collection = biocxml.load(f)
    with gzip.open(args.test_corpus, 'rt', encoding='utf8') as f:
        test_collection = biocxml.load(f)

    annotated_labels = sorted(set( anno.infons['label'] for doc in train_collection.documents+val_collection.documents+test_collection.documents for passage in doc.passages for anno in passage.annotations ))
    labels = ['O'] + [ f'{prefix}-{label}' for label in annotated_labels for prefix in ['B','I'] ]
    id2label = { idx:label for idx,label in enumerate(labels) }
    label2id = { label:idx for idx,label in enumerate(labels) }

    print(f"{id2label=}")

    base_model = 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract'
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    train_tokenized = make_dataset(train_collection, tokenizer, label2id)
    val_tokenized = make_dataset(val_collection, tokenizer, label2id)
    test_tokenized = make_dataset(test_collection, tokenizer, label2id)

    train_dataset = Dataset.from_list(train_tokenized)
    val_dataset = Dataset.from_list(val_tokenized)
    test_dataset = Dataset.from_list(test_tokenized)

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    if args.wandb_name:
        os.environ["WANDB_PROJECT"] = args.wandb_name

    tokenizer.save_pretrained(args.output_dir)
    
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=3,
        early_stopping_threshold=0.001
    )
    
    save_best_model_callback = SaveBestModelCallback(args.output_dir)
    
    training_args = TrainingArguments(
        output_dir="./tmp_ner_model",
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        metric_for_best_model="eval_macro_f1",
        load_best_model_at_end=True,
        greater_is_better=True,
        seed=42,
        num_train_epochs=32,
        report_to=("wandb" if args.wandb_name else "none")
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
        compute_objective=lambda metrics: metrics["eval_macro_f1"],
        n_trials=args.n_trials,
        direction="maximize",
    )

    print(best_trial)

    with open(f"{args.output_dir}/best_hyperparameters.json", "w") as f:
        json.dump(best_trial.hyperparameters, f, indent=2)

    # Load up the best model
    trainer.model = AutoModelForTokenClassification.from_pretrained(args.output_dir).to(trainer.args.device)
    
    train_report = run_classification_report(trainer, train_dataset, id2label, labels)
    val_report = run_classification_report(trainer, val_dataset, id2label, labels)
    test_report = run_classification_report(trainer, test_dataset, id2label, labels)

    with open(f"{args.output_dir}/train_report.txt", "w") as f:
        f.write(train_report)
    with open(f"{args.output_dir}/val_report.txt", "w") as f:
        f.write(val_report)
    with open(f"{args.output_dir}/test_report.txt", "w") as f:
        f.write(test_report)

    print("="*80)
    print("TEST REPORT:\n\n")
    print(test_report)
    print("="*80)
    print("Done.")
    
    
if __name__ == '__main__':
    main()

