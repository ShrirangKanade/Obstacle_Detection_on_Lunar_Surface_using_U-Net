from django.shortcuts import render
from django.contrib import messages

import matplotlib.pyplot as plt
import cv2 as cv
import torch
import torch.nn as nn
from PIL import Image
import numpy  as np
import torch.nn.functional as F
import torchvision.transforms.functional as TF


from django.core.files.storage import FileSystemStorage
import torchvision.transforms as transforms
import os
from os import path


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv=nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, 1,1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
#         nn.Dropout(0.5),

        nn.Conv2d(out_channels, out_channels, 3, 1,1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
#         nn.Dropout(0.5)
        )
    
        
    def forward(self, x):
        
        return self.conv(x)
    


class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64,128,256,512]):    #Features in the lists are from the arch
        super(UNET, self).__init__()
        self.ups=nn.ModuleList()     #creating modulelists to store operations output
        self.downs=nn.ModuleList()
        self.pool=nn.MaxPool2d(kernel_size=2, stride=2)   #preforms maxpooling to down sample the image size and extract features
        
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))   #assigining features to input channels after DoubleConv operation
            in_channels=feature                           #going downside in the architecture
                                                        
            
        for feature in reversed(features):               #reversing the features list for moving upward in architecture
            self.ups.append(nn.ConvTranspose2d(          #upsampling the image
                feature*2, feature, kernel_size=2, stride=2))   
            self.ups.append(DoubleConv(feature*2, feature))     #after upsampling performing DoubleConv operation 
            
        self.bottleneck=DoubleConv(features[-1],features[-1]*2)   #base layer creation
        self.final_conv=nn.Conv2d(features[0],out_channels, kernel_size=1)   #the Output Conv operartion

    def forward(self, x):           #the execution part of unet
        skip_connections=[]         #list to store skip_connections
        for down in self.downs:     #going downside in the architecture
            x=down(x)              
            skip_connections.append(x)     #store connections 
            x=self.pool(x)                 #maxpooling 
        x=self.bottleneck(x)               #base layer of unet
         
        skip_connections=skip_connections[::-1]     #reversing the skip connection list to move upwards
        
        for ind in range(0, len(self.ups), 2):
            x=self.ups[ind](x)
            skip_connection=skip_connections[ind//2]      #traversing the up module list to move upwards and storing connections
            
            if x.shape!=skip_connection.shape:
                x=TF.resize(x,size=skip_connection.shape[2:])   #resizing shapes for good divisibility operation
                
                
            concat_skip=torch.cat((skip_connection,x),dim=1)   #concating the skip connections to the up part of unet
            x=self.ups[ind+1](concat_skip)                     
        return self.final_conv(x)                              #returning the final conv output
    
    
model = UNET(in_channels=3, out_channels=4)

import base64
from .models import ImageX
from django.core.files.base import ContentFile
from io import BytesIO
device = torch.device('cpu')

def load_checkpoint(checkpoint, model):
    
     print("=> Loading checkpoint")
     model.load_state_dict(checkpoint["model_state_dict"])


# RUN THE MODEL_FILE.ipynb to generate model.pth and only use the best_model.pth
load_checkpoint(torch.load("E:\\UNET PYTORCH\\New folder\\projectApp - Copy - Copy\\Final_model.pth", map_location=device), model)


model.eval()

fs=FileSystemStorage()

def to_data_uri(pil_img):

    data = BytesIO()

    pil_img.save(data, "PNG")

    data64 = base64.b64encode(data.getvalue())

    return u'data:img/png;base64,'+data64.decode('utf-8')

def segment_image(request):
    print(request.FILES)
    if request.method == 'POST':
        
        # Get the uploaded image from the POST request
        uploaded_image = request.FILES['image']

        image = Image.open(uploaded_image)
        
        # Convert the image to RGB
        image_conv = image.convert('RGB')
        image = image.convert('RGB')
        imagex=image.resize((300,300))
        up_img = image
 
        transform = transforms.Compose([
            transforms.Resize((300,300),antialias=True),
        #     transforms.CenterCrop(256),
            transforms.ToTensor(),
        #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
       
        image = transform(image)
        print(type(image))
        Torchyman =image
        Torchyman = Torchyman.to(device)

        Preprocessed_img=Torchyman.unsqueeze(0)
#         #         # Process the image using the model
        with torch.no_grad():
        #     device =torch.device('cpu')
            test_out = model(Preprocessed_img)
            preds_x = torch.softmax(test_out,dim=1)
            predicted_mask = torch.argmax(preds_x,dim=1,keepdim=True) #predicted image by model#    
            numpy_image = predicted_mask.squeeze(0).squeeze(0).cpu().numpy()

        
        plt.switch_backend('AGG')
        plt.imshow(numpy_image)
        Get_flow = os.getcwd()
        print('CURRENT FLOW--> ',Get_flow)
        os.chdir(r"E:\UNET PYTORCH\New folder\projectApp - Copy - Copy\media\results")
        print('FLOW NOW--> ',os.getcwd())
        plt.axis('off')
        plt.savefig(f'{uploaded_image.name}',format='png',bbox_inches='tight',pad_inches = 0)
        print(numpy_image.shape) 
        width,height =numpy_image.shape
        plt.imshow(up_img.resize((width,height)))                                    # test img
        plt.imshow(numpy_image, alpha = 0.3)
        print('FLOW NOW--> ',f"{os.getcwd}+'\'+{uploaded_image.name}")
        plt.axis('off')
        plt.savefig(f'{uploaded_image.name}'+'trans.png',bbox_inches='tight',pad_inches = 0)
        joinx=path.join(os.getcwd(),uploaded_image.name)
        joinx_trans = path.join(os.getcwd(),uploaded_image.name+'trans.png')
        print("JOinx -->",joinx)
        output_map = Image.open(f"{joinx}")
        trans_map  = Image.open(f"{joinx_trans}")
        # if output_model.shape() != None:
        print('FLOW NOW--> ',Get_flow)
        os.chdir(Get_flow)
        
        en = ImageX()
        en.image = uploaded_image
     
        en.save()
  
       
        final_file1 = to_data_uri(output_map)
        final_file2=to_data_uri(trans_map)
        ds=fs.save(uploaded_image.name,uploaded_image)
        ds=fs.url(uploaded_image)

        # # Render the segmented image as a response
        return render(request, 'segmentation_result.html', {'uploaded_img': ds,'output': final_file1,'mix':final_file2})

    # return render(request, 'segmentation_form.html')
    return render(request, 'index.html')

