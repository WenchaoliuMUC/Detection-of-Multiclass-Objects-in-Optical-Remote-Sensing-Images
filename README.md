# Detection-of-Multiclass-Objects-in-Optical-Remote-Sensing-Images
Detection of Multiclass Objects in Optical Remote Sensing Images [GRSL paper](https://ieeexplore.ieee.org/document/8573851)  
# Running Environments:
Ubuntu 16.04
Anaconda([python 3.6](https://www.anaconda.com/download/#linux))   
torch 0.4.0a  [Pytorch](https://github.com/pytorch/pytorch#from-source)  
torchvision [Vision](https://github.com/pytorch/vision)
  
  
# Prerequisites
```python
  pip install shapely  
``` 
Download weight parameters form [Google Drive](https://drive.google.com/file/d/1pvsVnxh4fqhkiInegHLlqunLvCHJENz8/view?usp=sharing) and put it into backup file folder.  
Download [DOTA](https://captain-whu.github.io/DOTA/index.html) dataset  
  
Note: Please config the improved orientated response network follows [IORN](https://github.com/wdczs/ImprovedORN)

# Test:  
```python
  python detect.py  
```
# Train:  

```
  All training images of task 2 are divided into 1024×1024 pixel patches by the DOTA development kit  
```
```
  Generate train list of all divided images like "train_img_example/train.txt"   
```
```
  Change train list path in "cfg/dota.data"  
```
```python
  python train.py cfg/dota.data cfg/orn_4_dota.cfg backup/000057.weights  
```

# Evaluate for DOTA testset or valset:  
```python
  Generate test image list like "test_img/test_img_list.txt"  
```
```python
  Run "python test_for_map.py"
```

NOTE:1, This project is built based on the code [marvis](https://github.com/marvis/pytorch-yolo2). Thanks all the contributors. MIT License. 2, DOTA dataset was updated, we have not tested the new version. 3, If you can't find coresponding version of running environments, please contact me. 4, If you have any suggestions or questions, please contact me liuwenchaomuc@163.com.
