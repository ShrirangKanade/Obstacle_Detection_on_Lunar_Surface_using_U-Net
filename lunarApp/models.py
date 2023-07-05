from django.db import models

# Create your models here.

####################### Think about What Should code here #########################################################


from django.db import models

# Create your models here.


class ImageX(models.Model):
    image = models.ImageField(upload_to='images/')
    
  