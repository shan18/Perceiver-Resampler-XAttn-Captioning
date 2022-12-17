# Perceiver-Resampler-XAttn-Captioning

## Dependencies

Install all the project dependencies with

```
$ python3 -m pip install -r requirements.txt
```

## Dataset

The training scripts and the model architecture can support can image-to-text and video-to-text tasks right out of the box. What needs to be changed is the dataset class in `dataset.py` to support your own dataset. However, the current codebase offers the code to create data loaders only for the MSCOCO and the MLSLT (Multiligual Sign-Language Translation) dataset.

For instructions on how to download and setup the dataset, check the [README](data/README.md) in the `data/` directory.

## Training

For training a model with the default set of parameters, run

```
$ python run.py \
    --config-path=<Directory containing the config file> \
    --config-name=config.yaml \
    name=<name of the experiment> \
    mode=train \
    dataset.train_ds.visual_dir=<Directory containing the videos or images for train set> \
    dataset.train_ds.json_path=<Path to the json file with the transcripts for train set> \
    dataset.validation_ds.visual_dir=<Directory containing the videos or images for validation set> \
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
    dataset.test_ds.visual_dir=<Directory containing the videos or images for test set> \
    dataset.test_ds.json_path=<Path to the json file with the transcripts for test set> \
    trainer.exp_dir=<Directory to save the logs>
```
