from django.urls import path
from .views import RoomListView,BookingListView,RoomDetailView,CancelBookingView,BookingFormView,contact_us,about_us,gallery
app_name='hotel'
urlpatterns=[
    path('room_list/',RoomListView,name='RoomListView'),
    path('booking_list/',BookingListView.as_view(),name='BookingListView'),
    path('room/<category>',RoomDetailView.as_view(),name='RoomDetailView'),
    path('booking/cancel/<pk>',CancelBookingView.as_view(),name='CancelBookingView'),
    path('', BookingFormView, name='BookingFormView'),
    path('contact-us/', contact_us, name="contact_us"),
    path('about-us/', about_us, name="about_us"),
    path('gallery/',gallery,name='gallery')
]