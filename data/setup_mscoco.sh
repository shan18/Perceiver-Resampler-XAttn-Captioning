# Setup the directory for storing the dataset
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
mkdir -p $SCRIPT_DIR/mscoco
cd $SCRIPT_DIR/mscoco

# Download the dataset
wget http://images.cocodataset.org/zips/train2017.zip http://images.cocodataset.org/zips/val2017.zip http://images.cocodataset.org/annotations/annotations_trainval2017.zip

# Unzip the files
unzip -qq train2017.zip
unzip -qq val2017.zip
unzip -qq annotations_trainval2017.zip

# Keep only the relevant annotations
$ mv annotations/captions_train2017.json train2017.json
$ mv annotations/captions_val2017.json val2017.json

# Remove the unnecessary files
$ rm -rf annotations/ train2017.zip val2017.zip annotations_trainval2017.zip