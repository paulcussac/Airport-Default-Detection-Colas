import pandas as pd
import os
import torch
import numpy as np
import PIL

from datasets import load_dataset, Image, Dataset
from transformers import AutoFeatureExtractor, ViTFeatureExtractor,ViTForImageClassification,TrainingArguments, Trainer, BeitFeatureExtractor, TrainerCallback
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor)

from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import classification_report

from model_builder import *


# Load yaml config file.
with open("config.yml") as f:
    config = yaml.safe_load(f)


# Load data with image names associated with labels 
print("Loading data...", "/n")

os.chdir(config['ROOT_DIR_PATH']+'/model/dataset')
data = pd.read_csv('labels_train.csv')


# Converting images to dataset object
print("Converting images to dataset object...", "\n")

dataset = image_to_dataset(config['ROOT_DIR_PATH']+'/model/dataset/train')


# Label management
print("Label management...", "\n")

dataset = label_management(data_with_label=data, image_dataset=dataset)

id2label, label2id = label_id_converter(data_with_label=data)


# Defining feature extractor that will provide pixel data from the input dataset
print("Defining feature extractor...", "/n")

feature_extractor = feature_extractor(model=config['model'])


# Train Test split
print("Train-Test split...", "/n")

splits = dataset.train_test_split(test_size=config['val_size'])

train_ds = splits['train']
val_ds = splits['test']


# Data augmentation
print("Data augmentation...", "/n")

train_ds.set_transform(preprocess_train_ds)
val_ds.set_transform(preprocess_val_ds)


# Model building, using VIT for Image classification model (Hugging Face)
print("Model building...", "/n")

num_labels = len(id2label)

model = model( model=config['model'], 
               num_labels=num_labels, 
               problem_type="multi_label_classification", 
               use_auth_token=config['use_auth_token']
             )


# Redesigning the Hugging Face Trainer Class
print("Redesigning the Hugging Face Trainer Class...", "/n")

GLOBAL_SCORE_INDICES = range(0, len(id2label))

class MultiTaskClassificationTrainer(Trainer):
    def __init__(self, group_weights=None, **kwargs):
        super().__init__(**kwargs)
        
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs[0]
        
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits[:, GLOBAL_SCORE_INDICES], labels[:, GLOBAL_SCORE_INDICES])
        
        return (loss, outputs) if return_outputs else loss

class PrinterCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, logs=None, **kwargs):
        print(f"Epoch {state.epoch}: ")
        

# Training phase
print("Training Phase...", "/n")

# Defining arguments

training_args = TrainingArguments(
                                    model_checkpoint,
                                    remove_unused_columns=False,
                                    evaluation_strategy = "epoch",
                                    save_strategy = "epoch",
                                    learning_rate=5e-5,
                                    per_device_train_batch_size=config['batch_size'],
                                    gradient_accumulation_steps=4,
                                    per_device_eval_batch_size=config['batch_size'],
                                    num_train_epochs=config['num_epochs'],
                                    warmup_ratio=0.1,
                                    logging_steps=10,
                                    load_best_model_at_end=True,
                                    push_to_hub=False,
                                    metric_for_best_model="f1_macro"
                                )

# Defining trainer

trainer = MultiTaskClassificationTrainer(
                                            model=model,
                                            args=training_args,
                                            train_dataset=train_ds,
                                            eval_dataset=val_ds,
                                            data_collator=collate_fn,
                                            compute_metrics=compute_metrics,
                                            callbacks=[PrinterCallback]
                                        )

# Training 
trainer.train()


# Submission file
print("Submission phase..." + "/n")

# Load test set

test_dataset = image_to_dataset(config['ROOT_DIR_PATH'] + '/model/dataset/test')

submission_df = submission_df(test_dataset)

# Submission CSV file
os.chdir(config['ROOT_DIR_PATH'] + '/model')

submission_df.to_csv('submission.csv', index=False)