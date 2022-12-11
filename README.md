# Multilingual Sign Language Translation

## Dependencies

Install all the project dependencies with

```
$ python3 -m pip install -r requirements.txt
```

## Training

For training a model with the default set of parameters, run

```
$ python run.py \
    --config-path=<Directory containing the config file> \
    --config-name=config.yaml \
    name=<name of the experiment> \
    mode=train \
    dataset.train_ds.video_dir=<Directory containing the videos for train set> \
    dataset.train_ds.json_path=<Path to the json file with the transcripts for train set> \
    dataset.validation_ds.video_dir=<Directory containing the videos for validation set> \
    dataset.validation_ds.json_path=<Path to the json file with the transcripts for validation set> \
    trainer.exp_dir=<Directory to save the checkpoints and logs>
```

## Testing

To evaluate a trained model on a test set, run

```
HYDRA_FULL_ERROR=1 python run.py \
    --config-path=<Directory containing the config file> \
    --config-name=config.yaml \
    name=<name of the experiment> \
    mode=test \
    pretrained_name=<path to the checkpoint> \
    dataset.test_ds.video_dir=<Directory containing the videos for test set> \
    dataset.test_ds.json_path=<Path to the json file with the transcripts for test set> \
    trainer.exp_dir=<Directory to save the logs>
```
