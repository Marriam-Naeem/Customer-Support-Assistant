import os
import torch
import json
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from tqdm.auto import tqdm

# Import transformer components
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM, 
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig, 
    prepare_model_for_kbit_training, 
    get_peft_model,
    PeftModel
)
from datasets import Dataset

class LLMManager:
    """Manager for handling LLM operations."""
    
    def __init__(self,
                 model_name: str = "google/flan-t5-base",
                 model_type: str = "seq2seq",
                 use_lora: bool = True,
                 device: str = "auto",
                 output_dir: str = "model_output",
                 quantize: bool = True):
        """Initialize the LLM manager.
        
        Args:
            model_name: Name or path of the model to use
            model_type: Type of model ('causal' or 'seq2seq')
            use_lora: Whether to use LoRA for fine-tuning
            device: Device to use ('cpu', 'cuda', 'auto')
            output_dir: Directory for model outputs
            quantize: Whether to quantize the model for efficiency
        """
        self.model_name = model_name
        self.model_type = model_type
        self.use_lora = use_lora
        self.output_dir = output_dir
        self.quantize = quantize
        
        # Set device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load tokenizer and model
        self._load_tokenizer_and_model()
    
    def _load_tokenizer_and_model(self):
        """Load the tokenizer and model."""
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Ensure padding token is set for causal models
        if self.model_type == "causal" and self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Configure quantization if needed
        if self.quantize and self.device == "cuda":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0
            )
        else:
            quantization_config = None
        
        # Load model based on type
        try:
            if self.model_type == "causal":
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=quantization_config,
                    device_map="auto" if self.device == "cuda" else None,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                )
            else:  # seq2seq
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_name,
                    quantization_config=quantization_config,
                    device_map="auto" if self.device == "cuda" else None,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                )
            
            print(f"Successfully loaded {self.model_name}")
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
    
    def prepare_for_fine_tuning(self):
        """Prepare the model for LoRA fine-tuning."""
        if not self.use_lora:
            print("Not using LoRA. No preparation needed.")
            return
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=16,  # rank
            lora_alpha=32,
            target_modules=["q", "v"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM" if self.model_type == "causal" else "SEQ_2_SEQ_LM"
        )
        
        # Prepare model for kbit training if quantized
        if self.quantize and self.device == "cuda":
            self.model = prepare_model_for_kbit_training(self.model)
        
        # Get PEFT model
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
    
    def _format_banking_example(self, example: Dict[str, str]) -> str:
        """Format a banking example for training.
        
        Args:
            example: Dictionary with 'question' and 'answer' keys
            
        Returns:
            Formatted text for training
        """
        if self.model_type == "causal":
            return f"Question: {example['question']}\nAnswer: {example['answer']}\n"
        else:  # seq2seq
            return (example['question'], example['answer'])
    
    def _prepare_training_data(self, data: List[Dict[str, str]]):
        """Prepare data for training.
        
        Args:
            data: List of training examples
        """
        if self.model_type == "causal":
            # Format each example and tokenize
            formatted_data = [self._format_banking_example(ex) for ex in data]
            
            # Tokenize the data
            tokenized_data = self.tokenizer(
                formatted_data,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            # Create dataset
            dataset = Dataset.from_dict({
                "input_ids": tokenized_data["input_ids"],
                "attention_mask": tokenized_data["attention_mask"],
                "labels": tokenized_data["input_ids"].clone()
            })
            
        else:  # seq2seq
            # Split into inputs and targets
            inputs = [ex["question"] for ex in data]
            targets = [ex["answer"] for ex in data]
            
            # Tokenize inputs and targets separately
            tokenized_inputs = self.tokenizer(
                inputs,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt"
            )
            
            tokenized_targets = self.tokenizer(
                targets,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt"
            )
            
            # Create dataset
            dataset = Dataset.from_dict({
                "input_ids": tokenized_inputs["input_ids"],
                "attention_mask": tokenized_inputs["attention_mask"],
                "labels": tokenized_targets["input_ids"]
            })
        
        return dataset
    
    def fine_tune_on_banking_data(self, training_data: List[Dict[str, str]], epochs: int = 3, batch_size: int = 8):
        """Fine-tune the model on banking data.
        
        Args:
            training_data: List of training examples with 'question' and 'answer' keys
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        # Prepare model for fine-tuning if using LoRA
        if self.use_lora:
            self.prepare_for_fine_tuning()
        
        # Prepare training data
        dataset = self._prepare_training_data(training_data)
        
        # Set up training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            optim="adamw_torch",
            learning_rate=2e-4,
            weight_decay=0.01,
            logging_steps=10,
            save_steps=200,
            save_total_limit=3,
            fp16=self.device == "cuda",
        )
        
        # Create data collator
        if self.model_type == "causal":
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer, 
                mlm=False
            )
        else:
            data_collator = None
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
        )
        
        # Train
        trainer.train()
        
        # Save model
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        print(f"Model fine-tuned and saved to {self.output_dir}")
    
    def load_fine_tuned_model(self, model_path: str):
        """Load a fine-tuned model.
        
        Args:
            model_path: Path to the fine-tuned model
        """
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load model based on type
        if self.model_type == "causal":
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto" if self.device == "cuda" else None,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
        else:  # seq2seq
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_path,
                device_map="auto" if self.device == "cuda" else None,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
        
        print(f"Fine-tuned model loaded from {model_path}")
    
    def generate_response(self, 
                          query: str, 
                          context: Optional[str] = None, 
                          max_length: int = 512, 
                          temperature: float = 0.7) -> str:
        """Generate a response for a query.
        
        Args:
            query: User question
            context: Retrieved context (if any)
            max_length: Maximum response length
            temperature: Temperature for generation
            
        Returns:
            Generated response
        """
        # Format input based on context and model type
        if context:
            if self.model_type == "causal":
                input_text = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
            else:  # seq2seq
                input_text = f"Context: {context} Question: {query}"
        else:
            if self.model_type == "causal":
                input_text = f"Question: {query}\n\nAnswer:"
            else:  # seq2seq
                input_text = query
        
        # Tokenize input
        inputs = self.tokenizer(input_text, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=0.9,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        # Decode and return response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # For causal models, extract only the answer part
        if self.model_type == "causal" and "Answer:" in response:
            response = response.split("Answer:")[1].strip()
        
        return response

# Example usage
if __name__ == "__main__":
    # Initialize with a smaller model for testing
    llm_manager = LLMManager(
        model_name="google/flan-t5-base",
        model_type="seq2seq",
        use_lora=True,
        quantize=False  # Set to True for production
    )
    
    # Example banking QA pairs for fine-tuning
    training_data = [
        {"question": "How do I reset my password?", "answer": "You can reset your password by clicking on 'Forgot Password' on the login screen and following the steps."},
        {"question": "What is the daily ATM withdrawal limit?", "answer": "The daily ATM withdrawal limit for standard accounts is PKR 50,000."},
        # Add more examples...
    ]
    
    # Fine-tune (commented out for testing)
    # llm_manager.fine_tune_on_banking_data(training_data, epochs=1)
    
    # Test generation
    response = llm_manager.generate_response(
        query="How can I update my phone number?",
        context="To update phone number, go to 'Profile' section in the app and select 'Update Contact Information'."
    )
    print(f"Response: {response}")