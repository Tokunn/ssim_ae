#!/usr/bin/env python3
import os,sys
import glob
import subprocess

#classes = [os.path.basename(d) for d in glob.glob(os.path.expanduser('~/group/msuzuki/MVTechAD/*')) if os.path.isdir(d)]
#classes = ['capsule', 'carpet', 'metal_nut', 'cable']
classes = ['toothbrush']
print(classes)

NUMBER = 11
COMMENT = '128_MSELOSS_CropTrain_z500_color_batchnorm2_randomcrop'
IMGSIZE = 128
BATCHSIZE = 1024
#losses = ['MSE', 'SSIM']
#losses = ['SSIM']
losses = ['MSE']
EPOCHS = 300

os.makedirs("{:02}".format(NUMBER), exist_ok=True)

def main():
    with open('jobscript.sh', 'r') as f:
        jobscript = f.read()

    jobscripts = []
    for loss in losses:
        for aclass in classes:
            jsname = '{:02}/{}josbscript_{}_{}.sh'.format(NUMBER, loss, aclass, IMGSIZE)
            with open(jsname, 'w') as f:
                f.write(jobscript.format(loss, NUMBER, loss, aclass, COMMENT, aclass, BATCHSIZE, IMGSIZE, EPOCHS))
                jobscripts.append(jsname)
    #print(jobscripts)

    os.chdir('{:02}'.format(NUMBER))
    cmds = ["qsub -g gaa50088 ../{}".format(j) for j in jobscripts]
    for c in cmds:
        print(c)
    inp = input('[y/n] >')
    if inp=='y':
        for c in cmds:
            subprocess.call(c.split())

if __name__ == '__main__':
    main()
