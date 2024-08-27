import os
import time
import numpy as np
from transformers import (
    GPT2LMHeadModel, AutoTokenizer, Trainer, 
    TrainingArguments, GPTNeoForCausalLM, GPTJForCausalLM
)
from datasets import load_dataset, DatasetDict
from peft import get_peft_model, LoraConfig
#from evaluate import load as load_metric
from evaluate import load as evaluate_load
import torch
import pyreft
from transformers import DataCollatorForLanguageModeling
from nltk.translate.bleu_score import sentence_bleu
from difflib import SequenceMatcher
from Levenshtein import distance as levenshtein_distance
import numpy as np
from datasets import load_metric

os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


exact_match_metric = evaluate_load("exact_match", trust_remote_code=True)
f1_metric = evaluate_load("f1", trust_remote_code=True)
bleu_metric = evaluate_load("bleu", trust_remote_code=True)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    print(f"Sample Decoded Predictions: {decoded_preds[:5]}")
    print(f"Sample Decoded Labels: {decoded_labels[:5]}")

    decoded_preds = [pred.strip() for pred in decoded_preds if isinstance(pred, str)]
    decoded_labels = [label.strip() for label in decoded_labels if isinstance(label, str) and label]

    print(f"Stripped and Filtered Predictions: {decoded_preds[:5]}")
    print(f"Stripped and Filtered Labels: {decoded_labels[:5]}")

    if len(decoded_preds) == 0 or len(decoded_labels) == 0:
        return {
            "exact_match": 0.0,
            "f1": 0.0,
            "bleu": 0.0,
            "edit_distance": float('inf'),
            "token_accuracy": 0.0
        }
    if not all(isinstance(pred, str) for pred in decoded_preds):
        print("Non-string predictions detected")
    if not all(isinstance(label, str) for label in decoded_labels):
        print("Non-string labels detected")
    
    try:
        exact_match = exact_match_metric.compute(predictions=decoded_preds, references=decoded_labels)
    except Exception as e:
        exact_match = {"exact_match": 0.0}
        print(f"Error computing Exact Match score: {e}")
    try:
        f1 = f1_metric.compute(predictions=decoded_preds, references=decoded_labels, average="weighted")
    except Exception as e:
        f1 = {"f1": 0.0}
        print(f"Error computing F1 score: {e}")

    try:
        bleu = bleu_metric.compute(predictions=[pred.split() for pred in decoded_preds],
                                   references=[[label.split()] for label in decoded_labels])
        bleu_score = bleu['bleu']
    except Exception as e:
        bleu_score = 0.0
        print(f"Error computing BLEU score: {e}")

    edit_distances = [
        levenshtein_distance(pred, label)
        for pred, label in zip(decoded_preds, decoded_labels)
    ]
    avg_edit_distance = np.mean(edit_distances)

    token_accuracies = [
        np.mean([1 if pred_tok == label_tok else 0 for pred_tok, label_tok in zip(pred.split(), label.split())])
        for pred, label in zip(decoded_preds, decoded_labels)
    ]
    avg_token_accuracy = np.mean(token_accuracies)

    return {
        "exact_match": exact_match["exact_match"],
        "f1": f1["f1"],
        "bleu": bleu_score,
        "edit_distance": avg_edit_distance,
        "token_accuracy": avg_token_accuracy
    }


def load_model_and_tokenizer(model_name, cache_dir="./cache"):
    print(f"Loading model and tokenizer for {model_name}...")
    model_path = os.path.join(cache_dir, model_name.replace('/', '_'))
    if not os.path.exists(model_path):
        model = GPT2LMHeadModel.from_pretrained(model_name, cache_dir=cache_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        tokenizer.pad_token = tokenizer.eos_token
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
    else:
        model = GPT2LMHeadModel.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token

    
    print("Special tokens:", tokenizer.special_tokens_map)
    print("Tokenizer vocabulary size:", tokenizer.vocab_size)
    print("Model configuration:", model.config)
    return model, tokenizer

def subsample_dataset(dataset, fraction=1.0):
    print(f"Subsampling dataset with fraction: {fraction}")
    if "test" in dataset:
        small_test_dataset = dataset["test"].shuffle(seed=42).select(range(int(len(dataset["test"]) * fraction)))
    else:
        dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)
        small_test_dataset = dataset["test"].shuffle(seed=42).select(range(int(len(dataset["test"]) * fraction)))
    
    small_train_dataset = dataset["train"].shuffle(seed=42).select(range(int(len(dataset["train"]) * fraction)))
   
    print(f"Subsampled train size: {len(small_train_dataset)}")
    print(f"Subsampled test size: {len(small_test_dataset)}")
    return DatasetDict({"train": small_train_dataset, "test": small_test_dataset})

def preprocess_data(examples, tokenizer):
    inputs = [f"Translate the following question to SQL: {question}" for question in examples['question']]
    targets = [sql for sql in examples['query']]

   
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(text_target=targets, max_length=512, truncation=True, padding="max_length")["input_ids"]

    model_inputs["labels"] = labels
    return model_inputs

    
    print("Sample input text:", inputs[:2])
    print("Sample target SQL:", targets[:2])

    model_inputs = tokenizer(inputs, max_length=51, truncation=True, padding="max_length")
    
    
    print("Tokenized inputs:", model_inputs['input_ids'][:2])

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=51, truncation=True, padding="max_length")["input_ids"]

   
    print("Tokenized labels:", labels[:2])
    
    model_inputs["labels"] = labels
    return model_inputs

def fine_tune_model(model, tokenizer, ds_name, tokenized_dataset, model_name, training_args, strategy_name="Standard"):
    print(f"Starting fine-tuning for {model_name} on {ds_name} ({strategy_name})")
    trainer = Trainer(
        model=model.to(device),
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    start_time = time.time()
    trainer.train()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Training time for {model_name} on {ds_name} ({strategy_name}): {elapsed_time:.2f} seconds")
    results = trainer.evaluate()
    print(f"Evaluation results for {model_name} on {ds_name} ({strategy_name}):")
    print(results)
    return results


datasets = {
    "Spider": "xlangai/spider",
}

dataset_cache_dir = os.path.expanduser("~/.cache/huggingface/datasets/")
loaded_datasets = {}

for name, path in datasets.items():
    dataset_path = os.path.join(dataset_cache_dir, name)
    try:
        if os.path.exists(dataset_path):
            print(f"Loading {name} from cache.")
            ds = DatasetDict.load_from_disk(dataset_path)
        else:
            print(f"File not found at {dataset_path}. Loading from source.")
            ds = load_dataset(path)
            ds.save_to_disk(dataset_path)
        loaded_datasets[name] = ds
    except Exception as e:
        print(f"Error loading dataset {name}: {e}")
        continue  

gpt_models = {
    "gpt2": GPT2LMHeadModel,
    "gpt2-medium": GPT2LMHeadModel,
    "gpt2-large": GPT2LMHeadModel,
    "gpt2-xl": GPT2LMHeadModel,
    "gpt-neo": GPTNeoForCausalLM,
    "gpt-j": GPTJForCausalLM,
}

for model_name, model_class in gpt_models.items():
    print(f"Processing model: {model_name}")
    model, tokenizer = load_model_and_tokenizer(model_name)
    
    for ds_name, ds in loaded_datasets.items():
        print(f"Processing dataset: {ds_name}")
        small_dataset = subsample_dataset(ds)
        tokenized_dataset = small_dataset.map(lambda x: preprocess_data(x, tokenizer), batched=True)
        tokenized_dataset.set_format("torch")
        
        
        training_args = TrainingArguments(
            output_dir=f"./{model_name}-{ds_name}-finetuned-sql",
            evaluation_strategy="epoch",
            learning_rate=1e-4,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=10,
            weight_decay=0.01,
            save_total_limit=3,
            logging_dir=f'./logs-{model_name}-{ds_name}',
            gradient_accumulation_steps=2, 
        )
        
        fine_tune_model(model, tokenizer, ds_name, tokenized_dataset, model_name, training_args, "Standard")

        # LoRA 
        print(f"Applying LoRA to {model_name}")
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["attn.c_attn"],
            lora_dropout=0.1,
            fan_in_fan_out=True
            
        )
        model_lora = get_peft_model(model, lora_config)
        
        training_args_lora = TrainingArguments(
            output_dir=f"./lora_{model_name}-{ds_name}_finetuned_sql",
            evaluation_strategy="epoch",
            learning_rate=5e-4,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=10,
            weight_decay=0.01,
            save_total_limit=3,
            logging_dir=f'./logs-lora-{model_name}-{ds_name}',
        )
        
        fine_tune_model(model_lora, tokenizer, ds_name, tokenized_dataset, model_name, training_args_lora, "LoRA")

        # ReFT 
       #print(f"Applying ReFT to {model_name}")
        #reft_config = pyreft.ReftConfig(
            #representations=[{
                #"layer": l, 
                #"component": "block_output",
               # "low_rank_dimension": 2,
                #"intervention": pyreft.LoreftIntervention(
                    #embed_dim=model.config.hidden_size,
                   # low_rank_dimension=2
               # )
           # } for l in [2, 4, 6]]
       # )
       # reft_model = pyreft.get_reft_model(model, reft_config)
        #reft_model.set_device(device)
        
        #reft_training_args = TrainingArguments(
            #output_dir=f"./{model_name}-reft-{ds_name}-finetuned-sql",
            #evaluation_strategy="epoch",
            #learning_rate=5e-5,
            #per_device_train_batch_size=2,
            #per_device_eval_batch_size=2,
            #num_train_epochs=2,
            #weight_decay=0.01,
            #save_total_limit=3,
            #logging_dir=f'./logs-reft-{model_name}-{ds_name}',
        #)
        
        #fine_tune_model(reft_model, tokenizer, ds_name, tokenized_dataset, model_name, reft_training_args, "ReFT")
