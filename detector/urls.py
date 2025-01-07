from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('mediapipe/', views.mediapipe_detection, name='mediapipe'),
    path('yolo/', views.yolo_detection, name='yolo'),
    path('combined/', views.combined_detection, name='combined'),
    path('video_feed_mediapipe/', views.video_feed_mediapipe, name='video_feed_mediapipe'),
    path('video_feed_yolo/', views.video_feed_yolo, name='video_feed_yolo'),
    path('video_feed_combined/', views.video_feed_combined, name='video_feed_combined'),
]