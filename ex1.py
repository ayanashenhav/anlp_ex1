import os
from datasets import load_dataset, DatasetDict
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, \
    EvalPrediction
import evaluate
import numpy as np
import pandas as pd
import wandb
import fire


PRETRAINED_MODELS = ['google/electra-base-generator', 'bert-base-uncased',] # 'roberta-base', ]


def finetune_sst2_multiple(n_seeds: int = 3,
                           n_train_sample: int = -1,
                           n_validation_samples: int = -1,
                           n_test_samples: int = -1):
    dataset = load_dataset("sst2")
    dataset = slice_dataset(dataset, n_test_samples, n_train_sample, n_validation_samples)

    training_stats = list()
    train_time = 0
    best_model_accuracy = 0
    best_model_run_name = ""

    for model_name in PRETRAINED_MODELS:
        print(f"\n\n{model_name}\n\n")

        config = AutoConfig.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        tokenized_dataset = tokenize(dataset, tokenizer)

        for seed in range(n_seeds):
            model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config, )

            run_name = f"finetune_sst2_from_{model_name.replace('/', '_')}_seed_{seed}"
            print(f"\n{run_name}\n")

            wandb.init(project='anlp_ex1_sst2', entity='ayana-shenhav',
                       name=run_name, config={'model_name': model_name, 'seed': seed, },
                       reinit=True)

            trainer = Trainer(model=model,
                              args=TrainingArguments(output_dir=os.path.join(os.getcwd(), run_name),
                                                     report_to='wandb',
                                                     run_name=run_name,
                                                     save_strategy='no',
                                                     seed=seed,),
                              compute_metrics=compute_metrics,
                              train_dataset=tokenized_dataset['train'],
                              eval_dataset=tokenized_dataset['validation'],
                              tokenizer=tokenizer, )

            train_res = trainer.train()
            train_time += train_res.metrics['train_runtime']

            res = trainer.evaluate()
            training_stats.append(dict(run_name=run_name,
                                       model_name=model_name,
                                       seed=seed,
                                       eval_accuracy=res['eval_accuracy']))

            if res['eval_accuracy'] > best_model_accuracy:
                best_model_accuracy = res['eval_accuracy']
                best_model_run_name = trainer.args.run_name
                trainer.save_model("best_model")
                print(f"saved best model {trainer.args.run_name} in best_model dir, accuracy {res['eval_accuracy']}\n")

            wandb.finish()
            print(training_stats)

    df = pd.DataFrame(training_stats)
    print(df)

    best_model_trainer = get_best_model_trainer(dataset)
    print(f"\n\nsave prediction for best model - {best_model_run_name}\n\n")
    predict_time = save_predictions(best_model_trainer, tokenized_dataset)

    save_stats(df, predict_time, train_time)


def get_best_model_trainer(dataset):
    config = AutoConfig.from_pretrained('best_model')
    tokenizer = AutoTokenizer.from_pretrained('best_model')
    tokenized_dataset = tokenize(dataset, tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained('best_model', config=config, )
    trainer = Trainer(model=model,
                      args=TrainingArguments(output_dir=os.path.join(os.getcwd(), 'best_model'),
                                             save_strategy='no',),
                      compute_metrics=compute_metrics,
                      train_dataset=tokenized_dataset['train'],
                      eval_dataset=tokenized_dataset['validation'],
                      tokenizer=tokenizer, )
    res = trainer.evaluate()
    print(f"best model eval accuracy {res['eval_accuracy']}")
    return trainer


def save_stats(df, predict_time, train_time):
    mean = df.groupby('model_name')['eval_accuracy'].mean()
    std = df.groupby('model_name')['eval_accuracy'].std()
    with open("res.txt", 'w') as f:
        for model_name in PRETRAINED_MODELS:
            f.write(f"{model_name},{round(mean[model_name], 3)} +- {round(std[model_name], 3)}\n")
        f.write("----\n")
        f.write(f"train time,{train_time}\n")
        f.write(f"predict time,{predict_time}\n")


def save_predictions(best_model_trainer, tokenized_dataset):
    best_model_trainer.model.eval()
    best_model_trainer.args.per_device_eval_batch_size = 1
    predictions = best_model_trainer.predict(tokenized_dataset['test'].remove_columns(['sentence', 'idx', 'label']))
    predict_time = predictions.metrics['test_runtime']
    predictions = np.argmax(predictions.predictions, axis=1)
    with open('predictions.txt', 'w') as f:
        for sample, pred in zip(tokenized_dataset['test'], predictions):
            f.write(f"{sample['sentence']}###{pred}\n")
    return predict_time


def tokenize(dataset, tokenizer):
    def preprocess_function(examples):
        result = tokenizer(examples['sentence'], padding=False, truncation=True)
        return result

    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        desc="Running tokenizer on dataset",
    )
    return tokenized_dataset


def compute_metrics(eval_pred: EvalPrediction):
    metric = evaluate.load("accuracy")
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)


def slice_dataset(dataset, n_test_samples: int = -1, n_train_sample: int = -1, n_validation_samples: int = 1):
    dataset = DatasetDict(
        train=dataset['train'].select(range(n_train_sample)) if n_train_sample >= 0 else dataset['train'],
        validation=dataset['validation'].select(range(n_validation_samples)) if n_validation_samples >= 0 else dataset[
            'validation'],
        test=dataset['test'].select(range(n_test_samples)) if n_test_samples >= 0 else dataset['test'], )
    return dataset


if __name__ == '__main__':
    fire.Fire(finetune_sst2_multiple)