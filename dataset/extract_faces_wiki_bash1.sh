#extracting wiki images for folder 00 to 09
for ((ix=0; ix<=9; ix=ix+1))
do
    SOURCE_DIR=../../dataset/wiki/0
    SOURCE_DIR=$SOURCE_DIR$ix/
    #echo $SOURCE_DIR
    DEST_DIR=../../dataset/wiki/faces/0
    DEST_DIR=$DEST_DIR$ix/
    #echo $DEST_DIR
    python extract_faces.py --source_dir $SOURCE_DIR --save_dir $DEST_DIR&
done

