
# Obstacle Detection on Lunar Surface using U-Net✨

This repository represents a web app with a multi-class classification ML model which creates a segmented image of rocks and plain land.





## Description
* This project is developed to solve the problem of detecting obstacles (eg. rocks) on lunar surface.

* Implementation is based on the [U-Net architecture](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png) which creates a segmented image from raw image as an input.

## Dataset 

* [Artificial Lunar landscape dataset](https://www.kaggle.com/datasets/romainpessia/artificial-lunar-rocky-landscape-dataset) from Kaggle.


## Installation

### Requirements
- Python                    3.10.9 
- PyTorch                   1.12.1 (GPU)
- torchvision               0.13.1   
- OpenCV                    4.6.0 
- Django                    4.1.7  
- cudatoolkit               11.6.0              

Rest of the packages are listed in lunar packages list.txt file.
## Download the Obstacle Detection Model
- Download the " final_model.pth " file from following Drive [Link](https://drive.google.com/file/d/1WrvycZnVWwltSa6cjeTznEFOyNAwHEZu/view?usp=sharing).
- Download the file and add its path in views.py file in load_checkpoint() function.


## Deployment

To deploy this project open /lunarApp
/views.py  and run

```bash
  python manage.py runserver
```

It will launch the webapp, then follow below steps :
```bash
  1. Click on Choose File.
  2. Upload any file from Input samples eg PCAM1.png and click on segment.
  3. The results are displayed on new webpage.
```

