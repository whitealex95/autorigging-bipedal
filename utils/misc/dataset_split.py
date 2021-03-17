import os
import sys
import numpy as np

#BASE_DIR=os.path.dirname(os.path.abspath(__file__))
#sys.path.append(BASE_DIR)
#sys.path.append(os.path.join(BASE_DIR, 'data'))

action = ["Boxing", "Shoved", "Samba", "Walk"]
fCount = [66, 137, 547, 117]
owner = ['bjs', 'sht', 'kjh']

valid_num = 2

# validset generation
for i in range(valid_num):
    while(True):
        ownerIdx = np.random.randint(len(owner))
        modIdx = np.random.randint(1, 15+1)
        actionIdx = np.random.randint(len(action))
        frame = np.random.randint(fCount[actionIdx])
        
        filename = owner[ownerIdx]+str(modIdx).zfill(2)+"_"+action[actionIdx]+"_"+str(frame).zfill(3)+'.csv'

        if os.path.isfile('../data/joint22/train/transforms/'+owner[ownerIdx]+'/'+filename):
            print(filename + ' exists! go to validation set...')
            os.rename('../data/joint22/train/transforms/'+owner[ownerIdx]+'/'+filename, '../data/joint22/valid/transforms/'+filename)
            os.rename('../data/joint22/train/vertices/'+owner[ownerIdx]+'/'+filename, '../data/joint22/valid/vertices/'+filename)
            break

