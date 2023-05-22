import os
from datasets import load_dataset, DatasetDict
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, \
    EvalPrediction
import evaluate
import numpy as np
import pandas as pd
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

    training_stats = list()
    train_time = 0
    best_model_trainer = None
    best_model_accuracy = 0

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

            wandb.init(project='anlp_ex1_sentiment_analysis_full', entity='ayana-shenhav',
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

            train_res = trainer.train()
            train_time += train_res.metrics['train_runtime']

            res = trainer.evaluate()
            training_stats.append(dict(run_name=run_name,
                                       model_name=model_name,
                                       seed=seed,
                                       eval_accuracy=res['eval_accuracy']))
            # trainer.save_model(run_name)

            if res['eval_accuracy'] > best_model_accuracy:
                best_model_trainer = trainer
                best_model_accuracy = res['eval_accuracy']

            wandb.finish()

    df = pd.DataFrame(training_stats)

    # best_model_run_name = df.iloc[df['eval_accuracy'].idxmax()]['run_name']
    # model = AutoModelForSequenceClassification.from_pretrained(run_name)
    # tokenizer = AutoTokenizer.from_pretrained(run_name)
    print(f"best model: {best_model_trainer.args.output_dir}")
    best_model_trainer.model.eval()
    best_model_trainer.args.per_device_eval_batch_size = 1
    predictions = best_model_trainer.predict(tokenized_datasets['test'].remove_columns(['sentence', 'idx', 'label']))
    predict_time = predictions.metrics['test_runtime']
    predictions = np.argmax(predictions.predictions, axis=1)
    with open('predictions.txt', 'w') as f:
        for sample, pred in zip(dataset['test'], predictions):
            f.write(f"{sample['sentence']}###{pred}\n")

    mean = df.groupby('model_name')['eval_accuracy'].mean()
    std = df.groupby('model_name')['eval_accuracy'].std()
    with open("res.txt", 'w') as f:
        for model_name in pretrained_models:
            f.write(f"{model_name},{round(mean[model_name], 3)} +- {round(std[model_name], 3)}\n")
        f.write("----\n")
        f.write(f"train time,{train_time}\n")
        f.write(f"predict time,{predict_time}\n")


if __name__ == '__main__':
    fire.Fire(finetune_sst2_multiple)