import os
import re
from shutil import copyfile
import numpy as np
import cv2 as cv
from skimage import io

#root = '/Users/chenfenghan/Desktop/Projects/test'
root = '/Users/chenfenghan/Downloads/ADE20K_2016_07_26/images'
#dest = '/Users/chenfenghan/Desktop/Projects/dataset/'
dest = '/Users/chenfenghan/Desktop/Projects/test/'

seg_ext = 'seg'
part_ext = '_part'

objects_to_find = {
    'top': {'R': 100, 'G': 254},
    'bed': {'R': 0, 'G': 165}
}

# 'top': 'R': 100, 'G': 254, 'B':120
# 'top': 'R': 100, 'G': 254, 'B':30
# 'top': 'R': 100, 'G': 254, 'B':230

seg_class = {
    'work surface': {'R': 120, 'G': 15}
}

#person = {
    #'person': {'R': 70, 'G': 39}
#}

label_class = {
    'table': {'R': 100, 'G': 124},
    'desk': {'R': 20, 'G': 212},
    'coffee table': {'R': 20, 'G': 59},
    'billiard table': {'R': 70, 'G': 156}
}
# name[:18] for training file
# name[:16] for validation file

for subdir, dirs, files in os.walk(root):
    for file in files:
        name, _ = os.path.splitext(file)
        mask_to_save = np.zeros(shape=0)
        #person_mask = np.zeros(shape=0)

        if part_ext in name:
            mask_part = io.imread(os.path.join(root, subdir, file))
            #mask_part = cv.imread(os.path.join(root, subdir, file))
            #mask_part = cv.cvtColor(mask_part, cv.COLOR_BGR2RGB)

            mask_seg = io.imread(os.path.join(root, subdir, name[:16] + '_seg'+'.png'))
            #mask_seg = cv.cvtColor(mask_seg, cv.COLOR_BGR2RGB)

            mask_to_save = np.zeros(shape=mask_part.shape[:-1])


            for obj in objects_to_find.values():
                found_part = (mask_part[:, :, 0] == obj['R']) & (mask_part[:, :, 1] == obj['G'])   #True False

                instances = np.unique(mask_part[found_part][:, 2])

                for instance in instances:
                    part = (mask_part[:, :, 0] == obj['R']) & (mask_part[:, :, 1] == obj['G']) & (mask_part[:, :, 2] == instance)  # True False

                    obj_seg = mask_seg[part]   # RGB values
                    

                    for label in label_class.values():
                         obj_class = (obj_seg[:,  0] == label['R']) & (obj_seg[:,  1] == label['G'])    # True False
                         #print(obj_class)

                         if any(obj_class):
                            mask_to_save[part] = 255   # table top
                            break

            #if np.any(mask_to_save):
                # save the mask
                #print(f'Copying {name[:18]}')
                #cv.imwrite(dest + name[:18] + '_mask.png', mask_to_save)
                #pattern = re.compile('^' + name[:18]+'.jpg')
                #for to_copy in files:
                    #if pattern.match(to_copy):
                        #copyfile(os.path.join(root, subdir, to_copy),
                                 #os.path.join(dest, to_copy))


        if seg_ext in name:
            mask_se = io.imread(os.path.join(root, subdir, file))
            #mask_se = cv.cvtColor(mask_se, cv.COLOR_BGR2RGB)

            mask_to_save = np.zeros(shape=mask_se.shape[:-1])
            #person_mask = np.zeros(shape=mask_se.shape[:-1])

            for obj in seg_class.values():
                found = (mask_se[:, :, 0] == obj['R']) & (mask_se[:, :, 1] == obj['G'])  # True False
                #print(found)

                if np.any(found):
                    mask_to_save[found] = 255
                    #if np.any((mask_se[:, :, 0] == 70) & (mask_se[:, :, 1] == 39)):

                        #count_person = 0
                        #if count_person < 2000:
                            #count_person +=1
                            #person_mask[found] = 100  # person
                            #mask_to_save = mask_to_save + person_mask

                    #else:
                        #mask_to_save[found] = 255
                    break

        if np.any(mask_to_save):
            # save the mask
            isExist = os.path.exists(dest + name[:16] + '_mask.png')
            if isExist:
                mask_save = cv.imread(dest + name[:16] + '_mask.png')
                mask_save = cv.cvtColor(mask_save, cv.COLOR_BGR2GRAY)

                mask_save = mask_save + mask_to_save
                print(f'Copying {name[:16]}')
                cv.imwrite(dest + name[:16] + '_mask.png', mask_save)

            else:
                print(f'Copying {name[:16]}')
                cv.imwrite(dest + name[:16] + '_mask.png', mask_to_save)
                pattern = re.compile('^' + name[:16]+'.jpg')
                for to_copy in files:
                    if pattern.match(to_copy):
                        copyfile(os.path.join(root, subdir, to_copy),
                                os.path.join(dest, to_copy))

