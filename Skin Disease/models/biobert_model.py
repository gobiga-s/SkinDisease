"""
BioBERT Model Implementation for Text-based Skin Disease Prediction
Implements BioBERT for analyzing symptom descriptions and classifying skin conditions
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import logging
from typing import List, Dict, Tuple, Optional
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SymptomDataset(Dataset):
    """
    Custom dataset for symptom descriptions
    """
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize the text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class BioBERTSkinClassifier:
    """
    BioBERT-based classifier for skin disease prediction from symptom descriptions
    """
    
    def __init__(self, model_name: str = "dmis-lab/biobert-base-cased-v1.1", 
                 num_classes: int = 10, max_length: int = 512):
        """
        Initialize the BioBERT classifier
        
        Args:
            model_name (str): Pretrained BioBERT model name
            num_classes (int): Number of skin disease categories
            max_length (int): Maximum sequence length for tokenization
        """
        self.model_name = model_name
        self.num_classes = num_classes
        self.max_length = max_length
        self.tokenizer = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Skin disease categories
        self.class_names = [
            'Acne', 'Eczema', 'Psoriasis', 'Rosacea', 'Melanoma',
            'Basal Cell Carcinoma', 'Squamous Cell Carcinoma', 'Actinic Keratosis',
            'Dermatitis', 'Viral Infections'
        ]
        
        # Create label to class mapping
        self.label_to_class = {i: class_name for i, class_name in enumerate(self.class_names)}
        self.class_to_label = {class_name: i for i, class_name in enumerate(self.class_names)}
        
        logger.info(f"Initialized BioBERT classifier with {num_classes} classes")
        logger.info(f"Using device: {self.device}")
    
    def load_model(self, model_path: Optional[str] = None):
        """
        Load the BioBERT model and tokenizer
        
        Args:
            model_path (str): Path to fine-tuned model (optional)
        """
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path if model_path else self.model_name
            )
            
            # Load model configuration
            if model_path:
                # Load fine-tuned model
                self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
                logger.info(f"Loaded fine-tuned model from: {model_path}")
            else:
                # Load pretrained BioBERT and add classification head
                config = AutoConfig.from_pretrained(self.model_name, num_labels=self.num_classes)
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name, 
                    config=config
                )
                logger.info(f"Loaded pretrained model: {self.model_name}")
            
            # Move model to device
            self.model.to(self.device)
            
            logger.info("Model and tokenizer loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def prepare_data(self, data_file: str, text_column: str = 'symptoms', 
                    label_column: str = 'disease') -> Tuple[List[str], List[int]]:
        """
        Prepare training data from CSV file
        
        Args:
            data_file (str): Path to CSV file containing symptom descriptions
            text_column (str): Name of the column containing symptom text
            label_column (str): Name of the column containing disease labels
            
        Returns:
            tuple: (texts, labels)
        """
        try:
            # Load data
            df = pd.read_csv(data_file)
            
            # Extract texts and labels
            texts = df[text_column].astype(str).tolist()
            
            # Convert disease names to labels
            labels = [self.class_to_label.get(disease, 0) for disease in df[label_column]]
            
            logger.info(f"Loaded {len(texts)} samples from {data_file}")
            logger.info(f"Class distribution: {pd.Series(labels).value_counts().to_dict()}")
            
            return texts, labels
            
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            raise
    
    def create_datasets(self, texts: List[str], labels: List[int], 
                      test_size: float = 0.2, val_size: float = 0.1) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Create train, validation, and test datasets
        
        Args:
            texts (List[str]): List of symptom descriptions
            labels (List[int]): List of corresponding labels
            test_size (float): Proportion of data for testing
            val_size (float): Proportion of training data for validation
            
        Returns:
            tuple: (train_dataset, val_dataset, test_dataset)
        """
        try:
            from sklearn.model_selection import train_test_split
            
            # Split data into train+val and test
            train_texts, test_texts, train_labels, test_labels = train_test_split(
                texts, labels, test_size=test_size, random_state=42, stratify=labels
            )
            
            # Split train into train and validation
            train_texts, val_texts, train_labels, val_labels = train_test_split(
                train_texts, train_labels, test_size=val_size, random_state=42, stratify=train_labels
            )
            
            # Create datasets
            train_dataset = SymptomDataset(train_texts, train_labels, self.tokenizer, self.max_length)
            val_dataset = SymptomDataset(val_texts, val_labels, self.tokenizer, self.max_length)
            test_dataset = SymptomDataset(test_texts, test_labels, self.tokenizer, self.max_length)
            
            logger.info(f"Created datasets - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
            
            return train_dataset, val_dataset, test_dataset
            
        except Exception as e:
            logger.error(f"Error creating datasets: {str(e)}")
            raise
    
    def train_model(self, train_dataset: Dataset, val_dataset: Dataset, 
                   output_dir: str = "./biobert_skin_classifier", 
                   num_epochs: int = 3, batch_size: int = 16, 
                   learning_rate: float = 2e-5):
        """
        Fine-tune the BioBERT model
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            output_dir (str): Directory to save the fine-tuned model
            num_epochs (int): Number of training epochs
            batch_size (int): Training batch size
            learning_rate (float): Learning rate
        """
        try:
            # Define training arguments
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=num_epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                warmup_steps=500,
                weight_decay=0.01,
                logging_dir=f"{output_dir}/logs",
                logging_steps=100,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="eval_f1",
                greater_is_better=True,
                learning_rate=learning_rate,
                fp16=torch.cuda.is_available(),
                dataloader_num_workers=4,
                remove_unused_columns=False,
            )
            
            # Define metrics computation function
            def compute_metrics(eval_pred):
                predictions, labels = eval_pred
                predictions = np.argmax(predictions, axis=1)
                
                precision, recall, f1, _ = precision_recall_fscore_support(
                    labels, predictions, average='weighted'
                )
                accuracy = accuracy_score(labels, predictions)
                
                return {
                    'accuracy': accuracy,
                    'f1': f1,
                    'precision': precision,
                    'recall': recall
                }
            
            # Create trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=compute_metrics,
                tokenizer=self.tokenizer,
            )
            
            # Train the model
            logger.info("Starting model training...")
            trainer.train()
            
            # Save the model
            trainer.save_model(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            
            logger.info(f"Model training completed. Saved to: {output_dir}")
            
            return trainer
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise
    
    def predict(self, texts: List[str], return_probabilities: bool = True) -> Dict:
        """
        Make predictions on symptom descriptions
        
        Args:
            texts (List[str]): List of symptom descriptions
            return_probabilities (bool): Whether to return class probabilities
            
        Returns:
            Dict: Prediction results
        """
        try:
            if self.model is None or self.tokenizer is None:
                raise ValueError("Model must be loaded before prediction")
            
            self.model.eval()
            predictions = []
            probabilities = []
            
            with torch.no_grad():
                for text in texts:
                    # Tokenize input
                    inputs = self.tokenizer(
                        text,
                        truncation=True,
                        padding='max_length',
                        max_length=self.max_length,
                        return_tensors='pt'
                    )
                    
                    # Move to device
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    # Get model output
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    
                    # Get probabilities
                    probs = torch.nn.functional.softmax(logits, dim=-1)
                    
                    # Get predicted class
                    pred_class = torch.argmax(probs, dim=-1).item()
                    
                    predictions.append(pred_class)
                    probabilities.append(probs.cpu().numpy()[0])
            
            # Convert to class names
            predicted_classes = [self.label_to_class[pred] for pred in predictions]
            
            result = {
                'predictions': predictions,
                'predicted_classes': predicted_classes,
                'input_texts': texts
            }
            
            if return_probabilities:
                result['probabilities'] = probabilities
                result['class_probabilities'] = [
                    {self.class_names[i]: float(prob[i]) for i in range(len(self.class_names))}
                    for prob in probabilities
                ]
            
            return result
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise
    
    def predict_single(self, text: str, return_probabilities: bool = True) -> Dict:
        """
        Make prediction on a single symptom description
        
        Args:
            text (str): Symptom description
            return_probabilities (bool): Whether to return class probabilities
            
        Returns:
            Dict: Prediction result
        """
        result = self.predict([text], return_probabilities)
        
        single_result = {
            'predicted_class': result['predicted_classes'][0],
            'prediction': result['predictions'][0],
            'input_text': text
        }
        
        if return_probabilities:
            single_result['probabilities'] = result['probabilities'][0]
            single_result['class_probabilities'] = result['class_probabilities'][0]
            single_result['confidence'] = float(max(result['probabilities'][0]))
        
        return single_result
    
    def evaluate(self, test_dataset: Dataset) -> Dict:
        """
        Evaluate the model on test dataset
        
        Args:
            test_dataset: Test dataset
            
        Returns:
            Dict: Evaluation metrics
        """
        try:
            # Create trainer for evaluation
            training_args = TrainingArguments(
                output_dir="./temp_eval",
                per_device_eval_batch_size=16,
                logging_dir="./temp_eval/logs",
            )
            
            trainer = Trainer(
                model=self.model,
                args=training_args,
                eval_dataset=test_dataset,
                tokenizer=self.tokenizer,
            )
            
            # Evaluate
            results = trainer.evaluate()
            
            logger.info(f"Evaluation results: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            raise
    
    def save_model(self, save_path: str):
        """
        Save the fine-tuned model
        
        Args:
            save_path (str): Path to save the model
        """
        try:
            if self.model is None or self.tokenizer is None:
                raise ValueError("No model to save")
            
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            
            # Save class mapping
            with open(f"{save_path}/class_mapping.json", 'w') as f:
                json.dump({
                    'label_to_class': self.label_to_class,
                    'class_to_label': self.class_to_label,
                    'class_names': self.class_names
                }, f, indent=2)
            
            logger.info(f"Model saved successfully to: {save_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load_saved_model(self, model_path: str):
        """
        Load a saved fine-tuned model
        
        Args:
            model_path (str): Path to the saved model
        """
        try:
            # Load class mapping
            with open(f"{model_path}/class_mapping.json", 'r') as f:
                mapping = json.load(f)
                self.label_to_class = mapping['label_to_class']
                self.class_to_label = mapping['class_to_label']
                self.class_names = mapping['class_names']
            
            # Load model and tokenizer
            self.load_model(model_path)
            
            logger.info(f"Loaded saved model from: {model_path}")
            
        except Exception as e:
            logger.error(f"Error loading saved model: {str(e)}")
            raise


# Utility functions
def create_sample_data(output_file: str = "symptom_data.csv"):
    """
    Create sample symptom data for testing
    
    Args:
        output_file (str): Path to save the sample data
    """
    sample_data = [
        {"symptoms": "I have red, itchy patches on my arms that have been present for 2 weeks. The area is dry and sometimes flakes.", "disease": "Eczema"},
        {"symptoms": "Small red bumps with white heads on my forehead and cheeks, especially during stressful periods.", "disease": "Acne"},
        {"symptoms": "Thick, scaly, silver-colored patches on my elbows and knees that sometimes bleed when scratched.", "disease": "Psoriasis"},
        {"symptoms": "Persistent redness on my cheeks and nose, with small visible blood vessels and occasional bumps.", "disease": "Rosacea"},
        {"symptoms": "Dark irregular mole with uneven borders and multiple colors that has been growing in size.", "disease": "Melanoma"},
        {"symptoms": "Pearly bump on my nose that won't heal and occasionally bleeds.", "disease": "Basal Cell Carcinoma"},
        {"symptoms": "Red, scaly patch on my scalp that crusts over and sometimes bleeds.", "disease": "Squamous Cell Carcinoma"},
        {"symptoms": "Rough, sandpaper-like patches on my face and hands from sun exposure.", "disease": "Actinic Keratosis"},
        {"symptoms": "General skin inflammation with redness and itching all over my body.", "disease": "Dermatitis"},
        {"symptoms": "Small fluid-filled blisters on my lips that tingle before appearing.", "disease": "Viral Infections"}
    ]
    
    df = pd.DataFrame(sample_data)
    df.to_csv(output_file, index=False)
    logger.info(f"Sample data saved to: {output_file}")


if __name__ == "__main__":
    # Example usage
    classifier = BioBERTSkinClassifier()
    
    # Load pretrained model
    classifier.load_model()
    
    # Create sample data
    create_sample_data()
    
    # Prepare data
    texts, labels = classifier.prepare_data("symptom_data.csv")
    
    # Create datasets
    train_dataset, val_dataset, test_dataset = classifier.create_datasets(texts, labels)
    
    # Example prediction
    test_text = "I have red, itchy patches on my arms that are dry and flaky"
    result = classifier.predict_single(test_text)
    print(f"Prediction: {result}")
    
    # Example training (would need more data)
    # trainer = classifier.train_model(train_dataset, val_dataset)
    # evaluation_results = classifier.evaluate(test_dataset)
    # print(f"Evaluation: {evaluation_results}")
