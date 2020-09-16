import xml.etree.ElementTree as XET
import os
from os import mkdir, listdir
from os.path import splitext, exists, join


source_dir = [join("./images")]
target_dir = [join("./labels")]
custom_path = "data/custom/images"

valid_dir = join("data", "f01", "image")

sub_dir = ["train", "valid"]
out_dir = ["images", "labels"]

img_width = 500
img_height = 1200 



# label_idx x_center y_center width height

for dir in out_dir:
    if not exists(dir):
        mkdir(dir)


for dir in out_dir:
    for sub in sub_dir:
        if not exists(join(dir, sub)):
            mkdir(join(dir, sub))

for dir in ["xml"]:
    for file in sorted(listdir(f"{dir}/")):

        label_idx = []
        label_xmin = []
        label_ymin = []
        label_xmax = []
        label_ymax = []

        name = splitext(file)[0]

    
        tree = XET.parse(f"xml/{name}.xml")
        root = tree.getroot()

        for i in range(6, len(root)):
            
            x1, x2 =  int(root[i][4][0].text), int(root[i][4][2].text) 
            y1, y2 =  int(root[i][4][1].text), int(root[i][4][3].text) 
            width = int(root[i][4][2].text)-int(root[i][4][0].text) 
            height = int(root[i][4][3].text)-int(root[i][4][1].text) 
            
            if i == 6:
                f = open(f"object_detect_label/{name}.txt", 'w')
                f.write("{:d} {:d} {:d} {:d} {:d} {:d}\n".format(x1, y1, x2, y2, width, height) )
            else:
                f = open(f"object_detect_label/{name}.txt", 'a')
                f.write("{:d} {:d} {:d} {:d} {:d} {:d}\n".format(x1, y1, x2, y2, width, height) )
