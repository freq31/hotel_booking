from django.db import models
from django.conf import settings
from django.urls import reverse_lazy
import datetime
# Create your models here.
class Room(models.Model):
    ROOM_CATEGORIES=(
        ('SIN','SINGLE'),
        ('DBL','DOUBLE'),
        ('DEL','DELUXE'),
        ('KIN','KING'),
        ('QUE','QUEEN'),
        ('NAC','NON-AC'),
    )
    rate = models.FloatField()
    number =models.IntegerField()
    category=models.CharField(max_length=3,choices=ROOM_CATEGORIES)
    beds=models.IntegerField()
    capacity=models.IntegerField()
    def __str__(self):
        return f'room number: {self.number} ,category:{self.category} with {self.beds} for {self.capacity} people at rs {self.rate}/day'

class Booking(models.Model):
    user=models.ForeignKey(settings.AUTH_USER_MODEL,on_delete=models.CASCADE)
    room= models.ForeignKey(Room, on_delete=models.CASCADE)
    check_in=models.DateTimeField()
    check_out = models.DateTimeField()
    name = models.CharField(max_length=100)
    email = models.EmailField()
    number=models.IntegerField()
    total_price=models.FloatField()
    def __str__(self):
        return f'{self.name} has booked {self.room} from {self.check_in} to {self.check_out} \n total_price:{self.total_price}'

    def get_room_category(self):
        room_categories=dict(self.room.ROOM_CATEGORIES)
        room_category=room_categories.get(self.room.category)
        return room_category
    def get_cancel_booking_url(self):
        return reverse_lazy('hotel:CancelBookingView', args=[self.pk, ])
