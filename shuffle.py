import os
import random
out = open("/Users/LuLienHsi/Desktop/PyTorch-YOLOv3/data/custom/train_shuffle.txt",'w')
lines=[]
with open("/Users/LuLienHsi/Desktop/PyTorch-YOLOv3/data/custom/train.txt", 'r') as infile:
	for line in infile:
		lines.append(line)
random.shuffle(lines)
for line in lines:
	out.write(line)

out = open("/Users/LuLienHsi/Desktop/PyTorch-YOLOv3/data/custom/valid_shuffle.txt",'w')
lines=[]
with open("/Users/LuLienHsi/Desktop/PyTorch-YOLOv3/data/custom/valid.txt", 'r') as infile:
	for line in infile:
		lines.append(line)
random.shuffle(lines)
for line in lines:
	out.write(line)
