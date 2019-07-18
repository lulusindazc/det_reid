# HRI-market-project-A-Group: Det + Reid + Multi-Camera

This project contains the pedestrian detection and person re-identification code. Moreover, both pedestrian detection and re-identification are packaged in runcam*.py.

## Prerequisites

1. The packages required for each module need supplementing by relevant person in charge.

2. Python 2.7, torch 0.4.1

3. Run the code in Pycharm
 
## Camera Configurations

Four cameras are used in current version of market project. Specifically, 3 cameras are used for commodity detection, and one camera for checkout. The camera_id and IP of cameras are listed as follows.
```
a. runcam1.py----camera_id: 3 ----IP: 192.168.0.2----Commodity: Lao Ganma ```
```
b. runcam2.py----camera_id: 6 ----IP: 192.168.0.6----Commodity: Wine ```
```
c. runcam3.py----camera_id: 12 ----IP: 192.168.0.12----Commodity: Chips ```
```
d. runcam4.py----camera_id: 7 ----IP: 192.168.0.7----Checkout
```

## Run
1. Open a new terminal, and activate the ros environment
```
roscore
```
2. Start the ros publishing mechanism in Pycharm
```
../publish_image.py
```
3. collect the query dataset
```
cd ~/det_reid
python run_11-04.py
```
4. start the 4 cameras in market in Pycharm
```
runcam1.py/runcam2.py/runcam3.py/runcam4.py

## Time
@Creating Date: Nov 6, 2018
@Latest rectified: Nov 6, 2018



