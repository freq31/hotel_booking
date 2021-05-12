import datetime
from hotel.models import Room


def find_total_room_charge(check_in, check_out, category):
    days = check_out-check_in
    room_category = Room.objects.filter(category=category)
    room=room_category[0]
    total = days.days * room.rate
    return total
