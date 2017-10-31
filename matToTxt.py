from scipy import io
import os
import pdb

fileNamesPath = 'data/videogaze.mat'

matData = io.loadmat(fileNamesPath)
data = matData['videogaze'][0]

dataFolder = 'data'

train_list = []
test_list = []


train_file = 'train.txt'
test_file = 'test.txt'

def delFile(path):
    try:
       os.remove(path)
    except OSError:
       pass

delFile(train_file)
delFile(test_file)

trainWriter = open(train_file, 'w')
testWriter = open(test_file, 'w')
for i, entry in enumerate(data):
    print(str(i)+" ",)
    source_im_name, eyes, body_box, head_box, head_image, split, gaze_annot = entry
    source_im_name, eyes, body_box, head_box, head_image, split, gaze_annot = source_im_name[0], eyes[0], body_box[0], head_box[0], head_image[0], split[0], gaze_annot[0]

    for tgt_annot in gaze_annot:
        tgt_im_name, time, gaze = tgt_annot
        tgt_im_name, time, gaze = tgt_im_name[0], time[0], gaze[0]
        outputVals = [source_im_name, tgt_im_name, head_image, 0, eyes[0], eyes[1], gaze[0], gaze[1]]
        #  pdb.set_trace()
        outputEntry = " ".join([str(i) for i in outputVals]) + '\n'
        if split == 'train':
            train_list.append(outputEntry)
            trainWriter.write(outputEntry)
        else:
            test_list.append(outputEntry)
            testWriter.write(outputEntry)
print("train_list: %s, test_list: %s" % (len(train_list), len(test_list)))

trainWriter.close()
testWriter.close()
