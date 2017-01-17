import os
from random import shuffle


# If true, re-create all list files.
redo = True
# The root directory which holds all information of the dataset.
data_dir = '/home/prmct/Database/diabetic/messidor1_clear'
# The directory name which holds the image sets.
imgset_dir = "val"
# The direcotry which contains the images.
img_dir = "images"
img_ext = "jpg"
# The directory which contains the annotations.
anno_dir = "annotations"
anno_ext = "xml"

train_list_file = "./{}_ssd.txt".format(imgset_dir)

# Create training set.
# We follow Ross Girschick's split.
if redo or not os.path.exists(train_list_file):
    datasets = [imgset_dir]
    img_files = []
    anno_files = []
    for dataset in datasets:
        imgset_file = "{}/{}.txt".format(data_dir, dataset)
        print 'load from: ', imgset_file
        with open(imgset_file, "r") as f:
            for line in f.readlines():
                name = line.strip("\n")
                img_file = "{}/{}.{}".format(img_dir, name, img_ext)
                assert os.path.exists("{}/{}".format(data_dir, img_file)), \
                    "{}/{} does not exist".format(data_dir, img_file)
                anno_file = "{}/{}.{}".format(anno_dir, name, anno_ext)
                assert os.path.exists("{}/{}".format(data_dir, anno_file)), \
                    "{}/{} does not exist".format(data_dir, anno_file)
                img_files.append(img_file)
                anno_files.append(anno_file)
    # Shuffle the images.
    idx = [i for i in xrange(len(img_files))]
    shuffle(idx)
    with open(train_list_file, "w") as f:
        for i in idx:
            f.write("{} {}\n".format(img_files[i], anno_files[i]))
    print 'generating: ', train_list_file