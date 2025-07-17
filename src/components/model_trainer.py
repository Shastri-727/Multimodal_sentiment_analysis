from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
import pandas as pd
from sklearn.model_selection import train_test_split

# Load a standard dataset for demonstration (e.g., IMDB)
dataset = load_dataset("imdb")

# A smaller sample for faster fine-tuning
train_df, test_df = train_test_split(dataset['train'].to_pandas(), test_size=0.1)
small_train_dataset = Dataset.from_pandas(train_df.sample(n=1000)) # Using a small sample
small_eval_dataset = Dataset.from_pandas(test_df.sample(n=200))

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_train_dataset = small_train_dataset.map(tokenize_function, batched=True)
tokenized_eval_dataset = small_eval_dataset.map(tokenize_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch"
)

# Create a Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
)

# Start training
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./models/fine-tuned-bert")
tokenizer.save_pretrained("./models/fine-tuned-bert")

print("Model fine-tuning complete and saved.")