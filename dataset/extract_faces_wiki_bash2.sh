#extracting wiki images for folder 10 to 99
# please comment and uncomment the following line sequentially. 
#  we want to parallel run 10 program in one machine.
for ((ix=10; ix<=19; ix=ix+1))
#for ((ix=20; ix<=29; ix=ix+1))
#for ((ix=30; ix<=39; ix=ix+1))
#for ((ix=40; ix<=49; ix=ix+1))
#for ((ix=50; ix<=59; ix=ix+1))
#for ((ix=60; ix<=69; ix=ix+1))
#for ((ix=70; ix<=79; ix=ix+1))
#for ((ix=80; ix<=89; ix=ix+1))
#for ((ix=90; ix<=99; ix=ix+1))
do
    SOURCE_DIR=../../dataset/wiki/
    SOURCE_DIR=$SOURCE_DIR$ix/
    #echo $SOURCE_DIR
    DEST_DIR=../../dataset/wiki/faces/
    DEST_DIR=$DEST_DIR$ix/
    #echo $DEST_DIR
    python extract_faces.py --source_dir $SOURCE_DIR --save_dir $DEST_DIR&
done

