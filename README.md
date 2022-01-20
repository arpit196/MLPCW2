Steps for data preparation

1. wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
2. mkdir train
3. tar -xvf 'VOCtrainval_11-May-2012.tar' -C 'train'
4. Run the following in console, replace the path with your extracted path of pascalvoc dataset :!python create_pascal_tf_record.py --data_dir '/afs/inf.ed.ac.uk/user/s22/s2259628/mlpracFork/mlpractical/train/VOCdevkit/VOC2012' --image_data_dir '/afs/inf.ed.ac.uk/user/s22/s2259628/mlpracFork/mlpractical/train/VOCdevkit/VOC2012/JPEGImages' --label_data_dir '/afs/inf.ed.ac.uk/user/s22/s2259628/mlpracFork/mlpractical/train/VOCdevkit/VOC2012/SegmentationClass'
5. This will create a train and a val file in a folder named dataset, find its location and update the path '/home/arpit_manu6/dataset' in file Segment3.py at line with your tf_records path
