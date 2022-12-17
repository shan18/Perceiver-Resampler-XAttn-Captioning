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

For training a model with the hyperparamters present in `config.yaml`, run

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

## Evaluation

After training, you can use the command below to evaluate your models.

Don't feel like training a model yet? No worries, you can download a checkpoint that we trained using the default configuration present in `config.yaml` from [here](https://drive.google.com/file/d/11dYzjLq3nHtG0I_FP49YfztK0FOzbuca/view?usp=sharing).

To evaluate on the test set, run

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

*NOTE: The commands given above train or evaluate the model based on the default set of hyperparameters given in `config.yaml`. To tweak those values, please specify them explicitly via the command line.*
