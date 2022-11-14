This directory contains the dataset videos and the json files.

### Directory Structure

The data should have the following structure

```
data
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