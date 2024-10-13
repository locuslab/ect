
DATA_DIR = ./
DEST_DIR = ./

python dataset_tool.py convert --source=$DATA_DIR \
        --dest=$DEST_DIR/edm2-imagenet-64x64.zip --resolution=64x64 --transform=center-crop-dhariwal
