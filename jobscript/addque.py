import os,sys
import glob
import subprocess

#classes = [os.path.basename(d) for d in glob.glob(os.path.expanduser('~/group/msuzuki/MVTechAD/*')) if os.path.isdir(d)]
classes = ['capsule', 'carpet', 'metal_nut', 'cable']
print(classes)

NUMBER = 5
COMMENT = '256_16filter_z100_flipoff'
IMGSIZE = 256
BATCHSIZE = 512
losses = ['MSE', 'SSIM']

os.makedirs("{:02}".format(NUMBER), exist_ok=True)

def main():
    with open('jobscript.sh', 'r') as f:
        jobscript = f.read()

    jobscripts = []
    for loss in losses:
        for aclass in classes:
            jsname = '{:02}/{}josbscript_{}_{}.sh'.format(NUMBER, loss, aclass, IMGSIZE)
            with open(jsname, 'w') as f:
                f.write(jobscript.format(loss, NUMBER, loss, aclass, COMMENT, aclass, BATCHSIZE, IMGSIZE))
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
