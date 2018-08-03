import numpy as np

def getClasses(file="CamVid/label_colors.txt"):
    return bgr_numpy(loadClasses(file))

def loadClasses(file="CamVid/label_colors.txt"):
    classes = []
    data = open(file, 'r').read()
    lines = data.split('\n')
    for line in lines:
        if line == "":
            continue
        line = line.replace("		","\t")
        rgb,name = line.split('\t')
        r,g,b = rgb.split(' ')
        r,g,b = int(r),int(g),int(b)
        classes.append((name,(r,g,b)))
    return classes

def bgr_numpy(classes):
    for i in range(len(classes)):
        classes[i] = (classes[i][0], np.flip(np.asarray(classes[i][1]),0))
    return classes