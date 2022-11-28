# Multilingual Sign Language Translation

## Dependencies

Install all the project dependencies with

```
$ pip install -r requirements.txt
```

## Training

For training a model with the default set of parameters, run

```
$ python run.py \
    --config-name=config.yaml \
    name=<name of the experiment> \
    dataset.train_ds.video_dir=<Directory containing the videos for train set> \
    dataset.train_ds.json_path=<Path to the json file with the transcripts for train set> \
    dataset.validation_ds.video_dir=<Directory containing the videos for validation set> \
    dataset.validation_ds.json_path=<Path to the json file with the transcripts for validation set> \
    trainer.exp_dir=<Directory to save the checkpoints and logs>
```
