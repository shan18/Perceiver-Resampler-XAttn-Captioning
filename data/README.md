# Dataset

This directory should contain the dataset on which the model will be trained. The codebase uses json files to parse through the dataset and fetch the labels. The sections below list out the dataset setup instructions and the directory structure in which the COCO and the MLSLT dataset should be stored.

*NOTE: In the steps mentioned below, `$PROJ_ROOT_DIR` refers to the root directory of this repository i.e. if the repository was cloned at path `/some/path/Perceiver-Resampler-XAttn-Captioning` then this will be denoted by `$PROJ_ROOT_DIR`.*

## MSCOCO

The MSCOCO dataset contains approximately 125,000 images with a minimum of 5 captions per image covering a diverse range of objects, scenes and contexts. The dataset can be downloaded from [here](https://cocodataset.org/#download). To download and setup the dataset run

```
$ ./setup_mscoco.sh
```

### Directory Structure

The [MSCOCO](https://cocodataset.org/#download) dataset should have the following directory structure

```
$PROJ_ROOT_DIR/data
    └── mscoco
        ├── train2017
        |   ├── 000000479474.jpg
        |   ├── 000000230892.jpg
        |   ├── ...
        |   └── 000000206717.jpg
        ├── train2017.json
        ├── val2017
        |   ├── 000000357567.jpg
        |   ├── 000000140270.jpg
        |   ├── ...
        |   └── 000000163682.jpg
        └── val2017.json
```

## MLSLT

### Directory Structure

The MLSLT data should have the following structure

```
$PROJ_ROOT_DIR/data
    ├── train
    |   └── 10001
    |       ├── en.mp4
    |       ├── bg.mp4
    |       ├── ...
    |       └── zh.mp4
    |   ├── ...
    |   └── 9997
    |       ├── en.mp4
    |       ├── bg.mp4
    |       ├── ...
    |       └── zh.mp4
    ├── dev
    |   └── 10004
    |       ├── en.mp4
    |       ├── bg.mp4
    |       ├── ...
    |       └── zh.mp4
    |   ├── ...
    |   └── 9990
    |       ├── en.mp4
    |       ├── bg.mp4
    |       ├── ...
    |       └── zh.mp4
    ├── test
    |   └── 10006
    |       ├── en.mp4
    |       ├── bg.mp4
    |       ├── ...
    |       └── zh.mp4
    |   ├── ...
    |   └── 9998
    |       ├── en.mp4
    |       ├── bg.mp4
    |       ├── ...
    |       └── zh.mp4
    ├── train.json
    ├── dev.json
    └── test.json
```