import xml.etree.ElementTree as XET
import os   
from os import mkdir, listdir
from os.path import splitext, exists, join
import shutil

source_dir = [join("./images")]
target_dir = [join("./labels")]
custom_path = "data/custom/images"

sub_dir = ["train", "valid"]
out_dir = ["images", "labels"]

img_width = 500
img_height = 1200 

def delete(path):
    try:
        shutil.rmtree(path)
    except OSError as e:
        print(e)
    else:
        print("The directory is deleted successfully")

# label_idx x_center y_center width height


delete("./labels/train/")
delete("./labels/valid/")

for dir in out_dir:
    if not exists(dir):
        mkdir(dir)


for dir in out_dir:
    for sub in sub_dir:
        if not exists(join(dir, sub)):
            mkdir(join(dir, sub))

for dir in sub_dir:
    flag = True
    for file in sorted(listdir(f"./images/{dir}")):
        label_idx = []
        label_xmin = []
        label_ymin = []
        label_xmax = []
        label_ymax = []

        name = splitext(file)[0]

        path = join(custom_path, dir, f"{name}.png")
        
        if flag:
            f = open(f"{dir}.txt", 'w')
            f.write(f"{path}\n")
            flag = False
        else:
            f = open(f"{dir}.txt", 'a')
            f.write(f"{path}\n")

        tree = XET.parse(f"xml/{name}.xml")
        root = tree.getroot()

        for i in range(6, len(root)):
            
            x_center =  (float(root[i][4][0].text) + ( float(root[i][4][2].text)-float(root[i][4][0].text) )/2 ) 
            y_center =  (float(root[i][4][1].text) + ( float(root[i][4][3].text)-float(root[i][4][1].text) )/2 ) 
            width = float(root[i][4][2].text)-float(root[i][4][0].text) 
            height = float(root[i][4][3].text)-float(root[i][4][1].text) 

            x_center /= img_width
            y_center /= img_height
            width /= img_width
            height /= img_height
            if i == 6:
                f = open(f"labels/{dir}/{name}.txt", 'w')
                f.write("{:d} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(0, x_center, y_center, width, height) )
            else:
                f = open(f"labels/{dir}/{name}.txt", 'a')
                f.write("{:d} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(0, x_center, y_center, width, height) )
