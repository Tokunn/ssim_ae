from PIL import Image
import numpy as np
import os, sys, glob
import matplotlib.pyplot as plt

path = '/home/aca10370eo/group/msuzuki/MVTechAD'

classes = [d for d in glob.glob(os.path.join(path, '*')) if os.path.isdir(d)]
print(classes)

for aclass in classes:
    truthpath = os.path.join(aclass, 'ground_truth', 'good')
    testpath = os.path.join(aclass, 'test', 'good')

    testimglist = glob.glob(os.path.join(testpath, '*.png'))

    for testimgpath in testimglist:
        truthimgpath = os.path.join(truthpath, os.path.basename(testimgpath))

        testimg = Image.open(testimgpath)
        truthimg = Image.new('L', testimg.size)
        testimg.close()

        #truthimg.save(truthimgpath)
        print(truthimgpath)

    assert len(glob.glob(os.path.join(truthpath, '*.png'))) == len(glob.glob(os.path.join(testpath, '*.png')))
