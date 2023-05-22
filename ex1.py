import os
from datasets import load_dataset, DatasetDict
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, \
    EvalPrediction
import numpy as np
import evaluate
import wandb
import fire

#~/.cache/huggingface/datasets/


def finetune_sst2_multiple(n_seeds: int = 3,
                           n_train_sample: int = -1,
                           n_validation_samples: int = -1,
                           n_test_samples: int = -1):
    dataset = load_dataset("sst2")
    dataset = DatasetDict(train=dataset['train'].select(range(n_train_sample)) if n_train_sample >= 0 else dataset['train'],
                          validation=dataset['validation'].select(range(n_validation_samples)) if n_validation_samples >= 0 else dataset['validation'],
                          test=dataset['test'].select(range(n_test_samples)) if n_test_samples >= 0 else dataset['test'],)

    pretrained_models = ['bert-base-uncased', 'roberta-base', 'google/electra-base-generator']

    def compute_metrics(eval_pred: EvalPrediction):
        metric = evaluate.load("accuracy")
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return metric.compute(predictions=predictions, references=labels)

    for model_name in pretrained_models:
        print(model_name)

        config = AutoConfig.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        def preprocess_function(examples):
            result = tokenizer(examples['sentence'], padding=False, truncation=True)
            return result

        tokenized_datasets = dataset.map(
            preprocess_function,
            batched=True,
            desc="Running tokenizer on dataset",
        )

        for seed in range(n_seeds):
            model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config, )
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            run_name = f"finetune_sst2_from_{model_name.replace('/', '_')}_seed_{seed}"
            print(run_name)

            wandb.init(project='anlp_ex1_sentiment_analysis', entity='ayana-shenhav',
                       name=run_name, config={'model_name': model_name, 'seed': seed, },
                       reinit=True)

            trainer = Trainer(model=model,
                              args=TrainingArguments(output_dir=os.path.join(os.getcwd(), run_name),
                                                     report_to='wandb',
                                                     run_name=run_name,
                                                     save_strategy='no',
                                                     seed=seed,),
                              compute_metrics=compute_metrics,
                              train_dataset=tokenized_datasets['train'],
                              eval_dataset=tokenized_datasets['validation'],
                              tokenizer=tokenizer, )

            trainer.train()

            wandb.finish()


if __name__ == '__main__':
    fire.Fire(finetune_sst2_multiple)