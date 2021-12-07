from typing import Tuple
from django.db import models
from django.db.models.base import Model
from django.db.models.fields import AutoField, BooleanField, DateField, IntegerField
from django.db.models.fields.files import ImageField
# pip install pymysql 해야합니다.
# pip install mysqlclient 해야합니다.
###################################################################

class Mask_history(models.Model):
    camera_id = models.IntegerField() # primary_key ? -> 계속같은 번호가 들어올텐데 중복
    session_id = models.IntegerField()
    event_date = models.DateTimeField(auto_now=True) # 모델 save()시 자동으로 입력
    mask_detection = models.BooleanField(null=False) # 우리모델은 on / off 시에만 모델저장함
    img_path = models.CharField(max_length=255)

class Wash_history(models.Model):
    camera_id = models.IntegerField()
    session_id = models.IntegerField()
    event_date = models.DateTimeField(auto_now=True)
    step_1 = models.IntegerField(max_length=100,default=0)
    step_2 = models.IntegerField(max_length=100,default=0)
    step_3 = models.IntegerField(max_length=100,default=0)
    step_4 = models.IntegerField(max_length=100,default=0)

class Table_history(models.Model):
    camera_id = models.IntegerField()
    event_date = models.DateTimeField(auto_now=True)
    table_id = models.IntegerField()
    person_count = models.IntegerField() 
    alarm_detection = models.BooleanField()
    img_path = models.CharField(max_length=255, null=True)
