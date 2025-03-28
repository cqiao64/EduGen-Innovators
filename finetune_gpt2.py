import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW
import pandas as pd
import json
from tqdm import tqdm
import os

class MorphologyDataset(Dataset):
    def __init__(self, data_file, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        # Read CSV data
        df = pd.read_csv(data_file)
        
        # Process each question
        for _, row in df.iterrows():
            try:
                task = str(row['Task'])
                word = str(row['Word'])
                instruction = str(row['Instruction'])
                
                # Get choices
                choices = [
                    str(row['Choice_1']),
                    str(row['Choice_2']),
                    str(row['Choice_3'])
                ]
                
                # Get correct answer
                correct_answer = int(row['Correct_Answer']) - 1  # Convert to 0-based index
                
                # Build training text
                text = f"Task: {task}\n"
                text += f"Word: {word}\n"
                text += f"Question: {instruction}\n"
                text += "Choices:\n"
                for i, choice in enumerate(choices, 1):
                    text += f"{i}. {choice}\n"
                text += f"Correct Answer: {correct_answer + 1}\n"
                text += "---\n"
                
                # Ensure all text is string type
                text = text.replace('nan', '').replace('None', '')
                
                if len(text.strip()) > 0:  # Only add non-empty examples
                    self.examples.append(text)
                
            except Exception as e:
                print(f"Error processing row: {e}")
                continue

        print(f"Successfully loaded {len(self.examples)} training examples")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        text = self.examples[idx]
        
        # Ensure text is not empty
        if not text.strip():
            text = "Invalid example"
            
        # Encode text
        encodings = self.tokenizer(text, 
                                 truncation=True,
                                 max_length=self.max_length,
                                 padding='max_length',
                                 return_tensors='pt')
        
        return {
            'input_ids': encodings['input_ids'].squeeze(),
            'attention_mask': encodings['attention_mask'].squeeze()
        }

def train_model(model_name='gpt2', 
                data_file='Data/MC_data.csv',
                output_dir='models/finetuned_gpt2',
                learning_rate=5e-4,
                num_epochs=10,
                batch_size=4,
                temperatures=[0.7, 0.9, 1.1],
                training_steps=400):
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    # Prepare data
    dataset = MorphologyDataset(data_file, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    # Training loop
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()
    
    # Record training information
    training_info = {
        'model_name': model_name,
        'learning_rate': learning_rate,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'temperatures': temperatures,
        'training_steps': training_steps
    }
    
    with open(os.path.join(output_dir, 'training_info.json'), 'w') as f:
        json.dump(training_info, f, indent=2)
    
    global_step = 0
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        epoch_loss = 0
        
        for batch in tqdm(dataloader, desc=f"Training"):
            # Check if reached specified training steps
            if global_step >= training_steps:
                print(f"\nReached {training_steps} training steps. Stopping training.")
                break
                
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Forward pass
            outputs = model(input_ids=input_ids,
                          attention_mask=attention_mask,
                          labels=input_ids)
            
            loss = outputs.loss
            epoch_loss += loss.item()
            
            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            global_step += 1
            
            # Save model every 100 steps
            if global_step % 100 == 0:
                print(f"\nStep {global_step}: loss = {loss.item():.4f}")
                
                # Save current checkpoint
                checkpoint_dir = os.path.join(output_dir, f'checkpoint-{global_step}')
                os.makedirs(checkpoint_dir, exist_ok=True)
                model.save_pretrained(checkpoint_dir)
                tokenizer.save_pretrained(checkpoint_dir)
        
        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"\nEpoch {epoch+1} average loss: {avg_epoch_loss:.4f}")
        
        if global_step >= training_steps:
            break
    
    # Save final model
    final_model_dir = os.path.join(output_dir, 'final')
    os.makedirs(final_model_dir, exist_ok=True)
    model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    
    print(f"\nTraining completed. Model saved to {output_dir}")
    return model, tokenizer

if __name__ == "__main__":
    # Set training parameters
    training_params = {
        'model_name': 'gpt2',
        'data_file': 'Data/MC_data.csv',
        'output_dir': 'models/finetuned_gpt2',
        'learning_rate': 5e-4,
        'num_epochs': 10,
        'batch_size': 4,
        'temperatures': [0.7, 0.9, 1.1],
        'training_steps': 400
    }
    
    # Start training
    model, tokenizer = train_model(**training_params)
