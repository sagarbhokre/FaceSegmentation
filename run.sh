echo $1

if [ $1 == "city" ] ; then
    TRN_IMG_PATH=/home/sagar//Copilot/Segmentation/image-segmentation-keras/data/dataset1/images_prepped_train/
    TRN_ANN_PATH=/home/sagar//Copilot/Segmentation/image-segmentation-keras/data/dataset1/annotations_prepped_train/
    TST_IMG_PATH=/home/sagar//Copilot/Segmentation/image-segmentation-keras/data/dataset1/images_prepped_test/
    TST_ANN_PATH=/home/sagar//Copilot/Segmentation/image-segmentation-keras/data/dataset1/annotations_prepped_test/
    N_CLASS=10
    WIDTH=640
    HEIGHT=320
elif [ $1 == "local" ] ; then
    TRN_IMG_PATH=/home/sagar/datasets/FaceParsing/local/train/seg_images/
    TRN_ANN_PATH=/home/sagar/datasets/FaceParsing/local/train/seg_labels/
    TST_IMG_PATH=/home/sagar/datasets/FaceParsing/local/test/seg_images/
    TST_ANN_PATH=/home/sagar/datasets/FaceParsing/local/test/seg_labels/
    N_CLASS=2
    WIDTH=256
    HEIGHT=256
fi

EPOCH=$3

if [ $2 == "train" ] ; then
    echo "python  train.py  --save_weights_path=weights/ex1  --train_images=\"${TRN_IMG_PATH}\"  --train_annotations=\"${TRN_ANN_PATH}\"  --val_images=\"${TRN_IMG_PATH}\"  --val_annotations=\"${TRN_ANN_PATH}\"  --n_classes=$N_CLASS  --input_height=${HEIGHT}  --input_width=${WIDTH}  --model_name=\"fcn32\" --epochs 20 --optimizer_name=\"adamax\"" > exec_cmd
elif [ $2 == "predict" ] ; then
    export DISPLAY=:0
    echo "python  predict.py  --save_weights_path=weights/ex1  --test_images=\"${TST_IMG_PATH}\"  --n_classes=$N_CLASS  --input_height=${HEIGHT}  --input_width=${WIDTH}  --model_name=\"fcn32\" --epoch_number=${EPOCH}" > exec_cmd
elif [ $2 == "visualize" ] ; then
    export DISPLAY=:0
    echo "python  visualizeDataset.py  --images=\"${TRN_IMG_PATH}\"  --annotations=\"${TRN_ANN_PATH}\"  --n_classes=$N_CLASS" > exec_cmd
fi
. exec_cmd
exit
