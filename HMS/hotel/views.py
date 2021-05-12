from django.shortcuts import render,HttpResponse
from django.views.generic import  ListView,FormView,View,DeleteView
from .models import  Room,Booking
from hotel.booking_functions.availability import check_availability
from hotel.booking_functions.find_total_room_charge import find_total_room_charge
from .forms import AvailabilityForm
from django.urls import reverse,reverse_lazy

def BookingFormView(request):
    return render(request, 'booking_form.html')


def RoomListView(request):
    room=Room.objects.all()[0]
    room_categories= dict(room.ROOM_CATEGORIES)
    room_values=room_categories.values()
    room_list=[]
    for room_category in room_categories:
        room=room_categories.get(room_category)
        room_url=reverse('hotel:RoomDetailView',kwargs={'category':room_category})
        room_list.append((room,room_url))

    context={
        'room_list':room_list,
    }
    return render(request,'room_list_view.html',context)
class BookingListView(ListView):
    model = Booking
    template_name = "booking_list_view.html"
    def get_queryset(self,*args,**kwargs):
        if self.request.user.is_staff:
            booking_list=Booking.objects.all()
            return booking_list
        else:
            booking_list=Booking.objects.filter(user=self.request.user)
            return booking_list
class RoomDetailView(View):
    def get(self, request, *args, **kargs):
        category=self.kwargs.get('category',None)
        form=AvailabilityForm()
        room_list = Room.objects.filter(category=category)
        if len(room_list)>0:
            room=room_list[0]
            room_category=dict(room.ROOM_CATEGORIES).get(room.category,None)
            context = {
                'room_category': room_category,
                'form':form
            }
            return render(request, 'room_detail_view.html', context)
        else:
            return HttpResponse('category does not exist')




    def post(self, request, *args, **kargs):
        category = self.kwargs.get('category', None)
        room_list = Room.objects.filter(category=category)
        form=AvailabilityForm(request.POST)
        if form.is_valid():
            data=form.cleaned_data
        available_rooms = []
        for room in room_list:
            if check_availability(room, data['check_in'], data['check_out']): #and not(is_room_booked(room)):
                available_rooms.append(room)
        if len(available_rooms) > 0:
            room = available_rooms[0]
            total_price = find_total_room_charge(data['check_in'], data['check_out'], category)
            booking = Booking.objects.create(
                user=self.request.user,
                room=room,
                check_in=data['check_in'],
                check_out=data['check_out'],
                name=data['name'],
                email=data['email'],
                number=data['number'],
                total_price=total_price,
            )
            booking.save()

            return HttpResponse(booking)
        else:
            return HttpResponse('all rooms of this category are booked, plz try another one')



class CancelBookingView(DeleteView):
    model=Booking
    template_name = 'booking_cancel_view.html'
    success_url = reverse_lazy('hotel:BookingListView')
def contact_us(request):
    return render(request, 'contact_us.html')
def about_us(request):
    return render(request, 'about_us.html')
def gallery(request):
    room=Room.objects.all()[0]
    room_categories= dict(room.ROOM_CATEGORIES)
    room_values=room_categories.values()
    room_list=[]
    for room_category in room_categories:
        room=room_categories.get(room_category)
        room_url=reverse('hotel:RoomDetailView',kwargs={'category':room_category})
        room_list.append((room,room_url))

    context={
        'room_list':room_list,
    }
    return render(request,'gallery.html',context)