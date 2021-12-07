from io import BytesIO
import json
import re
from django.http.response import HttpResponse, HttpResponseRedirect, JsonResponse
from django.shortcuts import redirect, render
import requests
from django.views.decorators.csrf import csrf_exempt
from datetime import datetime, timedelta
from bs4 import BeautifulSoup as bs
from tensorflow.keras.preprocessing import image

from django.db.models import  Sum,Count, Max, Min, Avg
from .models import Mask_history,Wash_history,Table_history
from requests.sessions import session
import base64
from PIL import Image
import matplotlib.pyplot as plt
from django.core.files.storage import default_storage
from django.forms.models import model_to_dict
from django.contrib import messages

# global variables
cov_step = 1
cam_id=1
ses_id=1
bool_mask=True
WASH_NUMBER = 0
check_context = 0
table_context = {
    'data': 0
}
find_day = datetime.today()
load_day = find_day.strftime('%Y-%m-%d')

# Create your views here.
def index(request):
    print(" flask streaming ..  ")
    
    # index 불릴때마다 news crawling.
    try: 
        res = requests.get("https://www.ytn.co.kr/issue/corona.php")
        soup = bs(res.text, 'html.parser')
        news = soup.select('.newslist_wrap > div > ul > li')

        news_list = []
        i_count =0
        for i in news:
            i_count += 1
            n_title = i.select_one('.infowrap > .til').string
            n_time = i.select_one('.infowrap > .info > .date').string
            n_link = i.select_one('a').attrs['href']
            
            news_dic = {
                'index' : i_count,
                'title' :n_title,
                'time' :n_time,
                'link' :n_link,
            }
            news_list.append(news_dic)
    except Exception as e:
        print("except:",e)
    context = {
        'news_list' : news_list
    }
    
    return render(request, 'acto/index.html', context)

def mask(request):
   
    return render(
        request, 'mask/mask.html')

@csrf_exempt
def mask_on(request):
    now_time = datetime.today()
    print("마스크 착용, 현재시간 : ", now_time)
    # DB 저장부분
    global ses_id
    ses_id += 1
    
    data = Mask_history(camera_id=cam_id,session_id=ses_id,event_date=now_time,mask_detection=True)
    data.save()
    messages.info(request, 'mask-on')

    return render(
        request, 'mask/mask.html')

@csrf_exempt
def mask_off(request):
    now_time = datetime.today()
    cap_time = now_time.strftime('%y%m%d%H%M%S')

    if request.method == 'POST':
        file_img =  request.FILES['file']
        #print("******request",file_img)
        file_img.name = cap_time+".jpg"
        # path = '/static/NOMASK'+file_img.name
        default_storage.save("static/NOMASK/"+ file_img.name, file_img)

        # DB 저장부분
        global ses_id
        ses_id += 1
        data = Mask_history(camera_id=cam_id,session_id=ses_id,event_date=now_time,mask_detection=False,img_path="/static/NOMASK"+ '/' + cap_time+".jpg")
        data.save()
    
    print("마스크 미착용, 현재시간 : ", now_time)
    context = {
        'mask':False
    }
    return render(
        request, 'mask/mask.html', context)

@csrf_exempt
def wash(request):
    global WASH_NUMBER
    if request.method == "POST":
        # print("POST방식") 
        h_number = request.POST["hand_number"]
        print("수신된 num",h_number)
        WASH_NUMBER = h_number
        return render(
            request, 'wash/wash.html')
    else:
        return render(request, 'wash/wash.html')

@csrf_exempt
def wash_update(request):
    
    data = {
        'h_number' : WASH_NUMBER
    }
    print("html에서 요청 했음 ",data)
    return JsonResponse(data)

wash_session_id = 0
@csrf_exempt
def wash_save(request):
    #html -> views.py 로 보낸 json data 받기
    send_data = json.loads(request.body)

    #단계별로 진행 정도 받기
    
    step1 = int(send_data["hand1"])/10
    step2 = int(send_data["hand2"])/10
    step3 = int(send_data["hand3"])/10
    step4 = int(send_data["hand4"])/10
    #한 단계라도 5초이상 지속하지 않으면 데이터 저장 X
    if step1 >2 or step2 >2 or step3 >2 or step4 >2:
        #camera id 값
        camera_id = 1
        #현재 시간
        now_time = datetime.today()
        save_time = now_time.strftime('%y%m%d%H%M%S')
        #session_id 
        global wash_session_id
        wash_session_id +=1

        load_data = Wash_history(camera_id=camera_id,step_1=step1,step_2=step2,step_3=step3,step_4=step4, event_date = save_time,session_id = wash_session_id)
        load_data.save()
        print("저장완료",save_time)
    return render(request, 'wash/wash.html')

@csrf_exempt
def table(request):
    global cov_step
    global table_context
    temp = 0
    if request.method == 'POST':
        now_time = datetime.today()
        cap_time = now_time.strftime('%y%m%d%H%M%S')
        
        d = request.POST['data']
        file_img =  request.FILES['file']
        # 이미지 DB저장
        file_img.name = cap_time+".jpg"
        default_storage.save("static/CHECKTABLE/"+ file_img.name, file_img)
        d = int(d)
        
        if d > cov_step:
            if temp < d: 
                temp = d

                # DB save()
                now_time = datetime.today()
                data = Table_history(camera_id=cam_id,event_date=now_time,table_id=cov_step,person_count=d,alarm_detection=True,img_path="/static/CHECKTABLE"+ '/' + cap_time+".jpg")
                data.save()
                print("현재방역수칙 제한인원수 :",cov_step)
                print("테이블 인원 수:",d," 위반입니다!")
                table_context = {
                    'data': d
                }
    else:
        table_context['data'] = 0

    return render(
        request, 'table/table.html')

@csrf_exempt
def table_ajax(request):
    print("table-ajax!!!!")
    global table_context
    global check_context

    if table_context['data'] != check_context:
        check_context = table_context['data']
    else:
        table_context['data'] = 0
    return JsonResponse(table_context)
    
def stgs(request):
    global cov_step
    if request.method == 'POST':
        step_value = request.POST['step']

        #print(step_value)
        cov_step = int(step_value)

    context = {
            'cov_step' : cov_step
        }
    return render(request, 'acto/settings.html', context)

# <--------------------------------------------------------------->
# <-------------------------- 통계페이지 -------------------------->
# <--------------------------------------------------------------->
def statistics(request):
    return render(
        request, 'statistics/statistics.html')

def mask_history(request):
    if request.method == 'GET':

        # MASK obj
        mask_obj = Mask_history.objects.order_by('-id')

        # before 7 days 
        today_time = datetime.today()
        before1_time = today_time - timedelta(days=1)
        before2_time = today_time - timedelta(days=2)
        before3_time = today_time - timedelta(days=3)
        before4_time = today_time - timedelta(days=4)
        before5_time = today_time - timedelta(days=5)
        before6_time = today_time - timedelta(days=6)

        today_str = today_time.strftime('%Y%m%d')
        before1_str = before1_time.strftime('%Y%m%d')
        before2_str = before2_time.strftime('%Y%m%d')
        before3_str = before3_time.strftime('%Y%m%d')
        before4_str = before4_time.strftime('%Y%m%d')
        before5_str = before5_time.strftime('%Y%m%d')
        before6_str = before6_time.strftime('%Y%m%d')
        
        today_lst = []
        before1_lst = []
        before2_lst = []
        before3_lst = []
        before4_lst = []
        before5_lst = []
        before6_lst = []

        for i in mask_obj.values():
            obj_time = str(i['event_date'])[:10].replace('-','')
            #print(obj_time)
            if obj_time == today_str:
                today_lst.append(i)
            elif obj_time == before1_str:
                before1_lst.append(i)
            elif obj_time == before2_str:
                before2_lst.append(i)
            elif obj_time == before3_str:
                before3_lst.append(i)
            elif obj_time == before4_str:
                before4_lst.append(i)
            elif obj_time == before5_str:
                before5_lst.append(i)
            elif obj_time == before6_str:
                before6_lst.append(i)

        # T,F인원수 초기화
        today_t=b1_t=b2_t=b3_t=b4_t=b5_t=b6_t = 0
        today_f=b1_f=b2_f=b3_f=b4_f=b5_f=b6_f = 0

        # # # #
        for i in today_lst:
            if i['mask_detection'] == True:
                today_t +=1
            else:
                today_f +=1
        for i in before1_lst:
            if i['mask_detection'] == True:
                b1_t +=1
            else:
                b1_f +=1
        for i in before2_lst:
            if i['mask_detection'] == True:
                b2_t +=1
            else:
                b2_f +=1
        for i in before3_lst:
            if i['mask_detection'] == True:
                b3_t +=1
            else:
                b3_f +=1
        for i in before4_lst:
            if i['mask_detection'] == True:
                b4_t +=1
            else:
                b4_f +=1
        for i in before5_lst:
            if i['mask_detection'] == True:
                b5_t +=1
            else:
                b5_f +=1
        for i in before6_lst:
            if i['mask_detection'] == True:
                b6_t +=1
            else:
                b6_f +=1
        
        context = {
            'mask' : mask_obj,
            't_date':today_str,
            't_t':today_t,
            't_f':today_f,

            'b1_date':before1_str,
            'b1_t':b1_t,
            'b1_f':b1_f,

            'b2_date':before2_str,
            'b2_t':b2_t,
            'b2_f':b2_f,

            'b3_date':before3_str,
            'b3_t':b3_t,
            'b3_f':b3_f,

            'b4_date':before4_str,
            'b4_t':b4_t,
            'b4_f':b4_f,

            'b5_date':before5_str,
            'b5_t':b5_t,
            'b5_f':b5_f,

            'b6_date':before6_str,
            'b6_t':b6_t,
            'b6_f':b6_f,
        }
    print(context)
    return render(
        request, 'statistics/mask_history.html', context)

def wash_history(request):
    global load_day

    load_data = Wash_history.objects.all().order_by('-event_date')
    # avg_step1 = Wash_history.objects.filter(event_date__contains=load_day).aggregate(avg_step_1 = Avg("step_1"))['avg_step_1']
    # avg_step2 = Wash_history.objects.filter(event_date__contains=load_day).aggregate(avg_step_2 = Avg("step_2"))['avg_step_2']
    # avg_step3 = Wash_history.objects.filter(event_date__contains=load_day).aggregate(avg_step_3 = Avg("step_3"))['avg_step_3']
    avg_step41 = Wash_history.objects.filter(event_date__contains=load_day).aggregate(avg_step_4 = Avg("step_4"))['avg_step_4']
    avg_step1 = Wash_history.objects.all().aggregate(avg_step_1 = Avg("step_1"))['avg_step_1']
    avg_step2 = Wash_history.objects.all().aggregate(avg_step_2 = Avg("step_2"))['avg_step_2']
    avg_step3 = Wash_history.objects.all().aggregate(avg_step_3 = Avg("step_3"))['avg_step_3']
    avg_step4 = Wash_history.objects.all().aggregate(avg_step_4 = Avg("step_4"))['avg_step_4']

    total_count = len(load_data)
    
    print(load_data)

    #-------------------7일치 data---------------------------
    today_time = datetime.today()
    before1_time = today_time - timedelta(days=1)
    before2_time = today_time - timedelta(days=2)
    before3_time = today_time - timedelta(days=3)
    before4_time = today_time - timedelta(days=4)
    before5_time = today_time - timedelta(days=5)
    before6_time = today_time - timedelta(days=6)

    today_str = str(today_time.strftime('%Y-%m-%d'))
    before1_str = str(before1_time.strftime('%Y-%m-%d'))
    before2_str = str(before2_time.strftime('%Y-%m-%d'))
    before3_str = before3_time.strftime('%Y-%m-%d')
    before4_str = before4_time.strftime('%Y-%m-%d')
    before5_str = before5_time.strftime('%Y-%m-%d')
    before6_str = before6_time.strftime('%Y-%m-%d')

    today_str_change = str(today_time.strftime('%Y%m%d'))
    before1_str_change = str(before1_time.strftime('%Y%m%d'))
    before2_str_change = str(before2_time.strftime('%Y%m%d'))
    before3_str_change = before3_time.strftime('%Y%m%d')
    before4_str_change = before4_time.strftime('%Y%m%d')
    before5_str_change = before5_time.strftime('%Y%m%d')
    before6_str_change = before6_time.strftime('%Y%m%d')

    load_data_today = Wash_history.objects.filter(event_date__contains=today_str)
    load_data_before1 = Wash_history.objects.filter(event_date__contains=before1_str)
    load_data_before2 = Wash_history.objects.filter(event_date__contains=before2_str)
    load_data_before3 = Wash_history.objects.filter(event_date__contains=before3_str)
    load_data_before4 = Wash_history.objects.filter(event_date__contains=before4_str)
    load_data_before5 = Wash_history.objects.filter(event_date__contains=before5_str)
    load_data_before6 = Wash_history.objects.filter(event_date__contains=before6_str)

    today_avg_step1 = Wash_history.objects.filter(event_date__contains=today_str).aggregate(avg_step_1 = Avg("step_1"))['avg_step_1']
    today_avg_step2 = Wash_history.objects.filter(event_date__contains=today_str).aggregate(avg_step_2 = Avg("step_2"))['avg_step_2']
    today_avg_step3 = Wash_history.objects.filter(event_date__contains=today_str).aggregate(avg_step_3 = Avg("step_3"))['avg_step_3']
    today_avg_step4 = Wash_history.objects.filter(event_date__contains=today_str).aggregate(avg_step_4 = Avg("step_4"))['avg_step_4']

    before1_avg_step1 = Wash_history.objects.filter(event_date__contains=before1_str).aggregate(avg_step_1 = Avg("step_1"))['avg_step_1']
    before1_avg_step2 = Wash_history.objects.filter(event_date__contains=before1_str).aggregate(avg_step_2 = Avg("step_2"))['avg_step_2']
    before1_avg_step3 = Wash_history.objects.filter(event_date__contains=before1_str).aggregate(avg_step_3 = Avg("step_3"))['avg_step_3']
    before1_avg_step4 = Wash_history.objects.filter(event_date__contains=before1_str).aggregate(avg_step_4 = Avg("step_4"))['avg_step_4']

    before2_avg_step1 = Wash_history.objects.filter(event_date__contains=before2_str).aggregate(avg_step_1 = Avg("step_1"))['avg_step_1']
    before2_avg_step2 = Wash_history.objects.filter(event_date__contains=before2_str).aggregate(avg_step_2 = Avg("step_2"))['avg_step_2']
    before2_avg_step3 = Wash_history.objects.filter(event_date__contains=before2_str).aggregate(avg_step_3 = Avg("step_3"))['avg_step_3']
    before2_avg_step4 = Wash_history.objects.filter(event_date__contains=before2_str).aggregate(avg_step_4 = Avg("step_4"))['avg_step_4']
    print(before2_str,before2_avg_step1,before2_avg_step2,before2_avg_step3,before2_avg_step4)
    
    before3_avg_step1 = Wash_history.objects.filter(event_date__contains=before3_str).aggregate(avg_step_1 = Avg("step_1"))['avg_step_1']
    before3_avg_step2 = Wash_history.objects.filter(event_date__contains=before3_str).aggregate(avg_step_2 = Avg("step_2"))['avg_step_2']
    before3_avg_step3 = Wash_history.objects.filter(event_date__contains=before3_str).aggregate(avg_step_3 = Avg("step_3"))['avg_step_3']
    before3_avg_step4 = Wash_history.objects.filter(event_date__contains=before3_str).aggregate(avg_step_4 = Avg("step_4"))['avg_step_4']

    before4_avg_step1 = Wash_history.objects.filter(event_date__contains=before4_str).aggregate(avg_step_1 = Avg("step_1"))['avg_step_1']
    before4_avg_step2 = Wash_history.objects.filter(event_date__contains=before4_str).aggregate(avg_step_2 = Avg("step_2"))['avg_step_2']
    before4_avg_step3 = Wash_history.objects.filter(event_date__contains=before4_str).aggregate(avg_step_3 = Avg("step_3"))['avg_step_3']
    before4_avg_step4 = Wash_history.objects.filter(event_date__contains=before4_str).aggregate(avg_step_4 = Avg("step_4"))['avg_step_4']

    before5_avg_step1 = Wash_history.objects.filter(event_date__contains=before5_str).aggregate(avg_step_1 = Avg("step_1"))['avg_step_1']
    before5_avg_step2 = Wash_history.objects.filter(event_date__contains=before5_str).aggregate(avg_step_2 = Avg("step_2"))['avg_step_2']
    before5_avg_step3 = Wash_history.objects.filter(event_date__contains=before5_str).aggregate(avg_step_3 = Avg("step_3"))['avg_step_3']
    before5_avg_step4 = Wash_history.objects.filter(event_date__contains=before5_str).aggregate(avg_step_4 = Avg("step_4"))['avg_step_4']

    before6_avg_step1 = Wash_history.objects.filter(event_date__contains=before6_str).aggregate(avg_step_1 = Avg("step_1"))['avg_step_1']
    before6_avg_step2 = Wash_history.objects.filter(event_date__contains=before6_str).aggregate(avg_step_2 = Avg("step_2"))['avg_step_2']
    before6_avg_step3 = Wash_history.objects.filter(event_date__contains=before6_str).aggregate(avg_step_3 = Avg("step_3"))['avg_step_3']
    before6_avg_step4 = Wash_history.objects.filter(event_date__contains=before6_str).aggregate(avg_step_4 = Avg("step_4"))['avg_step_4']
    print(before6_str)

    context = {
        "wash" : load_data,
        "total_count" : total_count,
        "load_day" : load_day,
        "avg_step1" : avg_step1,
        "avg_step2" : avg_step2,
        "avg_step3" : avg_step3,
        "avg_step4" : avg_step4,

        "today_str" : today_str_change,
        "today_avg_step1": today_avg_step1,
        "today_avg_step2": today_avg_step2,
        "today_avg_step3": today_avg_step3,
        "today_avg_step4": today_avg_step4,

        "before1_str" : before1_str_change,
        "before1_avg_step1" : before1_avg_step1,
        "before1_avg_step2" : before1_avg_step2,
        "before1_avg_step3" : before1_avg_step3,
        "before1_avg_step4" : before1_avg_step4,

        "before2_str" : before2_str_change,
        "before2_avg_step1" : before2_avg_step1,
        "before2_avg_step2" : before2_avg_step2,
        "before2_avg_step3" : before2_avg_step3,
        "before2_avg_step4" : before2_avg_step4,
        
        "before3_str" : before3_str_change,
        "before3_avg_step1" : before3_avg_step1,
        "before3_avg_step2" : before3_avg_step2,
        "before3_avg_step3" : before3_avg_step3,
        "before3_avg_step4" : before3_avg_step4,

        "before4_str" : before4_str_change,
        "before4_avg_step1" : before4_avg_step1,
        "before4_avg_step2" : before4_avg_step2,
        "before4_avg_step3" : before4_avg_step3,
        "before4_avg_step4" : before4_avg_step4,

        "before5_str" : before5_str_change,
        "before5_avg_step1" : before5_avg_step1,
        "before5_avg_step2" : before5_avg_step2,
        "before5_avg_step3" : before5_avg_step3,
        "before5_avg_step4" : before5_avg_step4,

        "before6_str" : before6_str_change,
        "before6_avg_step1" : before6_avg_step1,
        "before6_avg_step2" : before6_avg_step2,
        "before6_avg_step3" : before6_avg_step3,
        "before6_avg_step4" : before6_avg_step4,
    }
    
    return render(
        request, 'statistics/wash_history.html', context)

def table_history(request):
    if request.method == 'GET':

        # table DB object
        table_obj = Table_history.objects.order_by('-id')
        context = {
            'table' : table_obj,
        }
    return render(
        request, 'statistics/table_history.html', context)
