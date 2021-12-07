"""config URL Configuration
The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('mask/', views.mask, name='mask'),
    path('mask/on/', views.mask_on, name='mask_on'),
    path('mask/off/', views.mask_off, name='mask_off'),
    path('wash/', views.wash, name='wash'),
    path('wash_update/', views.wash_update, name='wash_update'),
    path('wash_save/', views.wash_save, name='wash_save'),
    path('table/', views.table, name='table'),
    path('table_ajax/', views.table_ajax, name='table_ajax'),
    path('settings/', views.stgs, name='stgs'),
    path('statistics/', views.statistics, name='statistics'),
    path('statistics/mask/', views.mask_history, name='mask_history'),
    path('statistics/wash/', views.wash_history, name='wash_history'),
    path('statistics/table/', views.table_history, name='table_history'),
    #path('<str:room_name>/', views.room, name='room'),
]