from PIL import Image
from numpy import asarray
from glob import glob
import os
# load the image and convert into
# numpy array
empty_images = glob('data/generated_path/empty*')
generated_images = glob('data/a_star*')
def clean():
    for img_name in empty_images:

        img = Image.open(img_name)  
        numpydata = asarray(img)
        h,w = numpydata.shape[:2]

        for i in range(h):
            for j in range(w):

                if numpydata[i,j][0] != 0 or numpydata[i,j][1] != 0 or numpydata[i,j][2] != 0:
                    numpydata[i,j] = [255,255,255]
        
        img = Image.fromarray(numpydata, 'RGB')
        filename = f'data/cleaned_empty/{img_name.split("/")[-1]}'
        print(filename)
        img.save(filename)


def remove():
    for img in generated_images:
        os.remove(img)


remove()