#Path for file 
# source_path = "/media/kwaugh/RAID/Documents/visual_recognition/experiment/data/videogaze_images"
source_path = "data/videogaze_images"
# source_path = "data/videogaze_images"
# face_path = "/media/kwaugh/RAID/Documents/visual_recognition/experiment/data/videogaze_heads"
face_path = "data/videogaze_heads"
# face_path = "data/videogaze_heads"
target_path = "target"

#Train and test input files. Format is described in README.md 
test_file = "test_flag.txt"
train_file = "train_flag.txt"

#Training parameters
workers = 4;
epochs = 30
batch_size = 100

base_lr = 0.0001
momentum = 0.9
weight_decay = 1e-4
print_freq = 10
prec1 = 0
best_prec1 = 0
lr = base_lr

side_w = 20
