from django import forms
class AvailabilityForm(forms.Form):
    """"
    ROOM_CATEGORIES = (
        ('SIN', 'SINGLE'),
        ('DBL', 'DOUBLE'),
        ('DEL', 'DELUXE'),
        ('KIN', 'KING'),
        ('QUE', 'QUEEN'),
        ('NAC', 'NON-AC'),
    )
    room_category=forms.ChoiceField(choices=ROOM_CATEGORIES,required=True)
    """
    check_in=forms.DateTimeField(required=True,input_formats=["%d-%m-%YT%H:%M",])
    check_out = forms.DateTimeField(required=True, input_formats=["%d-%m-%YT%H:%M", ])
    name=forms.CharField(required=True)
    number=forms.IntegerField(required=True)
    email=forms.EmailField(required=True)