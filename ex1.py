from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

if __name__ == '__main__':

    dataset = load_dataset("sst2")

    pretrained_models = ['bert-base-uncased', 'roberta-base', 'google/electra-base-generator']

    args = TrainingArguments(max_steps=2200)

    for model_name in pretrained_models:
        print(model_name)

        config = AutoConfig.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)


        def preprocess_function(examples):
            # Tokenize the texts
            result = tokenizer(examples['sentence'], padding=False, truncation=True)
            return result


        tokenized_datasets = dataset.map(
            preprocess_function,
            batched=True,
            desc="Running tokenizer on dataset",
        )

        trainer = Trainer(model=model,
                          train_dataset=tokenized_datasets['train'],
                          eval_dataset=tokenized_datasets['validation'],
                          tokenizer=tokenizer, )

        trainer.train()