
# Obstacle Detection on Lunar Surface using U-Net ✨

This repository represents a web app with a multi-class classification ML model which creates a segmented image of rocks and plain land.





## 📄 Description
* This project is developed to solve the problem of detecting obstacles (eg. rocks) on lunar surface.

* Implementation is based on the [U-Net architecture](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png) which creates a segmented image from raw image as an input.

## 📁 Dataset 

* [Artificial Lunar landscape dataset](https://www.kaggle.com/datasets/romainpessia/artificial-lunar-rocky-landscape-dataset) from Kaggle.


## 🛠 Installation

### Requirements
- Python                    3.10.9 
- PyTorch                   1.12.1 (GPU)
- torchvision               0.13.1   
- OpenCV                    4.6.0 
- Django                    4.1.7  
- cudatoolkit               11.6.0              

Rest of the packages are listed in lunar packages list.txt file.
## 👁Download the Obstacle Detection Model
- Download the " final_model.pth " file from following Drive [Link](https://drive.google.com/file/d/1WrvycZnVWwltSa6cjeTznEFOyNAwHEZu/view?usp=sharing).
- Download the file and add its path in views.py file in load_checkpoint() function.


## 🖥 Deployment
- Install the dependencies locally.

- To deploy this project open /lunarApp/views.py  and run :

```bash
  python manage.py runserver
```

- It will launch the webapp, then follow below steps :

  1. Click on Choose File.
  2. Upload any file from Input samples eg PCAM1.png and click on segment.
  3. The results are displayed on new webpage.🎉🎊


## 🧠 Hyperparameters

| Hyperparameters             | Values                                                              |
| ----------------- | ------------------------------------------------------------------ |
| Epoch  | 30  |
| Batch Size | 16|
| Learning Rate | 0.0001|
| Optimizer | Adam |
| Scheduler | ReduceLROnPlateau |
| Accuracy | IoU|
| Loss Function | Cross Entropy Loss|






## 📷 Screenshot
![Finaloutput](https://github.com/ShrirangKanade/Obstacle_Detection_on_Lunar_Surface_using_U-Net/assets/110344056/8c183d67-d35c-4d69-bdff-c174b43073d4)




## 📄 Published Papers
- [Paper 1](https://ijrpr.com/uploads/V3ISSUE12/IJRPR8857.pdf)
- [Paper 2](https://ijrpr.com/uploads/V4ISSUE5/IJRPR12979.pdf)



## 😇 Feedback

If you have any feedback, please reach out to us at coder.shrirang.kanade@gmail.com

