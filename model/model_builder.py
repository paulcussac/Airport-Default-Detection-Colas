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



def image_to_dataset(path_start):
    """    
    Input : path to image folder
        
    This function takes all images and gathers them in a dataset under PIL type.
    """
    # Storing all the paths to images in a dict
    list_path = [path_start + filename for filename in os.listdir('dataset/train')]
    path_dict = {"image":list_path}

    # Converting the dict to a dataset object
    dataset = Dataset.from_dict(path_dict).cast_column("image", Image())
    
    return dataset


def label_management(data_with_label, image_dataset):
    """
    Input : 
            - data_with_label : Dataframe with image names associated with its labels.
            - image_dataset   : Image Dataset defined by image_to_dataset function.
        
    This function modifies the image_dataset to add to each image its respective labels.
    """
    # Gathers the five labels in a list and stock it in a new column named label.
    data['label'] = data.apply(lambda x: [x.FISSURE, x.REPARATION, x['FISSURE LONGITUDINALE'], x.FAÏENCAGE, x['MISE EN DALLE']] , axis=1)
    
    # Create the column of labels to be added to the dataset
    column_of_labels = []
    
    for i in range(dataset.shape[0]):
        filename = list_path[i][65:]
        row_dataset = data[data.filename == filename]
        list_label = list(row_dataset['label'])[0]
        list_label = np.array(list_label, dtype = np.float32).tolist()
        column_of_labels.append(list_label)
        
    dataset = dataset.add_column(name="label", column=column_of_labels)
    
    return dataset


def label_id_converter(data_with_label):
    """
    Input : Dataframe with image names associated with its labels.
        
    This function creates two dictionaries associating labels to ids and vice versa. Returns first label2id, then id2label.
    """
    # Collecting labels in a list
    all_labels = data_with_label.columns.drop("filename").to_list()
    
    # Defining converter dictionaries
    id2label = {k:l for k, l in enumerate(all_labels)}
    label2id = {l:k for k, l in enumerate(all_labels)}
    
    return id2label, label2id


def feature_extractor(model):
    """
    Input : pre-trained transformer encoder model from which to fine-tune
        
    This function generates a feature extractor to resize and normalize images for the model.
    """
    # Defining feature extractor
    feature_extractor = ViTFeatureExtractor.from_pretrained(model)
    
    return feature_extractor


def data_augmentation(train_ds, val_ds, feature_extractor):
    """
    Input : 
            - dataset           : dataset.
            - feature_extractor : Feature extractor defined with feature_extractor function.
    
    This function performs data augmentation on train set and validation set.
        - Train set      :  
        - Validation set :
    """
    
    
    return train_ds, val_ds


def model(model, num_labels, problem_type, use_auth_token):
    """
    Input : 
            - model          : pre-trained transformer encoder model from which to fine-tune.
            - num_labels     : number of labels.
            - problem_type   : For Multi-label classification, use "multi_label_classification".
            - use_auth_token : Token to connect to Hugging Face API.
    
    This function builds the model
    """
    model = ViTForImageClassification.from_pretrained(
                                                        model,
                                                        num_labels = num_labels,
                                                        problem_type = problem_type,
                                                        ignore_mismatched_sizes = True,
                                                        use_auth_token = use_auth_token
                                                        )
    
    return model


def collate_fn(examples):
    
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    
    return {"pixel_values": pixel_values, "labels": labels}


def get_preds_from_logits(logits):
    
    ret = np.zeros(logits.shape)
    
    # We fill 1 to every class whose score is higher than some threshold
    # In this example, we choose that threshold = 0.0
    ret[:, GLOBAL_SCORE_INDICES] = np.array(logits[:, GLOBAL_SCORE_INDICES] >= 0.0).astype(int)
    
    return ret


def compute_metrics(eval_pred):
    
    logits, labels = eval_pred
    final_metrics = {}
    
    # Deduce predictions from logits
    predictions = get_preds_from_logits(logits)

    # The global f1_metrics
    final_metrics["f1_micro"]  = f1_score(labels, predictions, average="micro")
    final_metrics["f1_macro"]  = f1_score(labels, predictions, average="macro")
    final_metrics["f1_weight"] = f1_score(labels, predictions, average="weighted")
    
    # Classification report
    print("Classification report for global scores: ")
    print(classification_report(labels[:, GLOBAL_SCORE_INDICES], predictions[:, GLOBAL_SCORE_INDICES], zero_division=0))
    return final_metrics    


def preprocess_train_ds(example_batch):
    
    normalize = Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)

    dataset_transforms = Compose(
                                    [
                                        Resize(feature_extractor.size),
                                        ToTensor(),
                                        normalize,
                                    ]
                                )
    example_batch["pixel_values"] = [dataset_transforms(image.convert("RGB")) for image in example_batch["image"]]
    
    return example_batch 

    
def preprocess_val_ds(example_batch):
        
    normalize = Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)   
    
    dataset2_transforms = Compose(
        [
            Resize(feature_extractor.size),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )
    
    example_batch["pixel_values"] = [dataset2_transforms(image.convert("RGB")) for image in example_batch["image"]]
    
    return example_batch    
    

def return_pred(filepath):
                
    image_to_decode = PIL.Image.open(filepath)
    encoding = feature_extractor(image_to_decode.convert("RGB"), return_tensors="pt")
    outputs = model(**encoding)
    logits = outputs.logits
    preds = get_preds_from_logits(logits)
                
    return preds


def submission_df(test_dataset):
    
    test_dataset['filepath']     = test_dataset.apply(lambda x: os.getcwd() + "/test/" + x.filename,axis=1)
    
    test_dataset['predictions']  = test_dataset.apply(lambda x: return_pred(x.filepath),axis=1)
    
    test_dataset["FISSURE"]               = test_dataset.apply(lambda x: int(x.predictions.tolist()[0][0]), axis=1)
    test_dataset["REPARATION"]            = test_dataset.apply(lambda x: int(x.predictions.tolist()[0][1]), axis=1)
    test_dataset["FISSURE LONGITUDINALE"] = test_dataset.apply(lambda x: int(x.predictions.tolist()[0][2]), axis=1)
    test_dataset["FAÏENCAGE"]             = test_dataset.apply(lambda x: int(x.predictions.tolist()[0][3]), axis=1)
    test_dataset["MISE EN DALLE"]         = test_dataset.apply(lambda x: int(x.predictions.tolist()[0][4]), axis=1)
    
    submission = test_dataset.drop(columns=['filepath','predictions'], axis=1)
    
    return submission
    