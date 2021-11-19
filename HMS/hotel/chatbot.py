# Import packages
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import tensorflow as tf
import nltk
nltk.download('punkt')
from nltk.stem.porter import PorterStemmer

from datetime import datetime
import random
import string
import pandas as pd

# Get the GPU device name
device_name = tf.test.gpu_device_name()

# The device name should look like the following:
if device_name == '/device:GPU:0':
    print('Found GPU at: {}'.format(device_name))
else:
    raise SystemError('GPU not available')

# If there's a GPU available...
if torch.cuda.is_available():

    # Tell PyTorch to use the GPU
    device = torch.device("cuda")

    print('There are {} GPU(s) available.'.format(torch.cuda.device_count()))

    print('GPU name:', torch.cuda.get_device_name(0))

# If not, use the CPU
else:
    print('GPU not found, using the CPU instead.')
    device = torch.device("cpu")

# Create a dictionary with your training data
intents = {
    'intents': [
        {
            'tag': 'greeting',
            'patterns': [
                'Hi',
                'Hey',
                'Hello',
                'How are you',
                'Is anyone there?',
                'Good day'
            ],
            'responses': [
                'Hey',
                'Hello, thanks for visiting',
                'Hi there, what can I do for you?',
                'Hi there, how can I help?'
            ]
        },
        {
            'tag': 'goodbye',
            'patterns': [
                'Bye',
                'See you later',
                'Goodbye'
            ],
            'responses': [
                'See you later, thanks for visiting',
                'Have a nice day',
                'Bye! Come back again soon'
            ]
        },
        {
            'tag': 'thanks',
            'patterns': [
                'Thanks',
                'Thank you',
                "That's helpful",
                'Thanks a lot!'
            ],
            'responses': [
                'Happy to help!',
                'Anytime!',
                "My pleasure"
            ]
        },
        {
            'tag': 'booking',
            'patterns': [
                'I would like to make a reservation',
                'Can I book a room?',
                "I want to make a booking"
            ],
            'responses': [
                'Sure, I only need to know a few details',
                "Lovely. Let's begin with the reservation"
            ]
        },
        {
            'tag': 'cancellation',
            'patterns': [
                'I would like to cancel my booking',
                'Can I cancel a room?',
                "I want to make a cancellation",
                'Cancel a booking'
            ],
            'responses': [
                'Sure, which is your reference number?',
                "I only need to know the reference number of your booking"
            ]
        },
        {
            'tag': 'payments',
            'patterns': [
                'When do I have to pay?',
                'Can I pay by card?',
                "Do you accept cash?",
                'When is the payment?'
            ],
            'responses': [
                'The payment is on arrival. We accept cash and card',
                'You can pay on arrival by cash or card'
            ]
        },
        {
            'tag': 'location',
            'patterns': [
                'Where is the hotel located?',
                'Can you give me the address?',
                "I want to know the exact location"
            ],
            'responses': [
                'The Hollywood Hotel is located at Vine St, Los Angeles, CA 99999',
                'You will find us at Vine St, Los Angeles, CA 99999'
            ]
        },
        {
            'tag': 'contacts',
            'patterns': [
                'I would like to contact the marketing department',
                'I want to speak with someone from sales',
                "Can I have the email of finance?"
            ],
            'responses': [
                'Sure. To contact with them, send an email to:',
                'Of course. The email to contact with the department is:'
            ]
        }
    ]
}
stemmer = PorterStemmer()


def tokenize(sentence):
    '''
    This function takes a sentence as an input,
    and returns a list of its tokens
    '''
    return nltk.word_tokenize(sentence)


def bag_of_words(tokenized_sentence, all_words):
    '''
    Function to represent a sentence into a vector of float numbers
    input: list of tokens in a sent and a list of all the words in the text
    output: vector equal to the vocab length for each sentence
    '''
    tokenized_sentence = [stemmer.stem(w.lower()) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)

    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0

    return bag

all_words = []
tags = []
xy = []

# Save all the keywords in different variables
for intent in intents['intents']:
    tag= intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

# Create the vocabulary
ignore_words = ['?', '!', '.', ',']
all_words = [stemmer.stem(w.lower()) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

# Set the final training data
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)


class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        # no activation and no softmax
        return out

# Hyperparameters
batch_size=8
hidden_size=8
output_size=len(tags)
input_size = len(X_train[0])
learning_rate = 0.001
num_epochs = 1000

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2)

model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device)

        # Forward
        outputs = model(words)
        loss = criterion(outputs, labels)

        # Backward and optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'epoch {epoch + 1}/{num_epochs}, loss = {loss.item():.4f}')

print(f'final loss, loss = {loss.item():.4f}')

all_bookings = pd.DataFrame()


class Customer:
    rate = 99.00

    def __init__(self, name, dates, room, service):
        self.name = name
        self.dates = (datetime.strptime(dates[0], '%d/%m/%Y').date(), datetime.strptime(dates[1], '%d/%m/%Y').date())
        self.room = room
        self.service = service

    def ID(self):
        letters = string.ascii_uppercase
        digits = string.digits
        a = random.sample(letters, 3) + random.sample(digits, 4)
        self.id = ''.join(a)

    def nights(self):
        nights = (self.dates[1] - self.dates[0]).days
        return nights

    def final_price(self):
        price = self.rate * float(self.nights())
        return price

    def __str__(self):
        return f'''
        > Mr./Miss. {self.name[1]}, 
        >
        > We are delighted to confirm your booking with us for the {self.dates[0]} till the {self.dates[1]}. 
        > A {self.room} with {self.service} for the final rate of £{self.rate} per night. 
        > Total price: £{self.final_price()}
        > Your reference number is {self.id}. 
        > Keep this number in case you want to modify or cancel your booking in the future.
        >
        > Best,
        > The Hollywood Hotel
        '''
departments = {
    'marketing': ['marketing', 'seo', 'community manager'],
    'sales': ['reservations', 'sales', 'booking'],
    'accountancy': ['accountancy', 'finance', 'purchase']
}

def contact_dept(user_sent, departments):
  '''
  Takes the sentence and all the departments as input
  and returns the department email that the user wants to be contact with.
  '''
  email = None
  for k,v in departments.items():
    for d in user_sent:
      if d in v:
        email = f'{k}@hollywoodhotel.com'
  return email

model_dic = model.state_dict()
model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_dic)
model.eval()

bot_name = 'Bot '
print("Let's chat: type 'quit' to exit")

while True:
  sentence = input('You: ')
  if sentence == 'quit':
    break

  sentence = tokenize(sentence)
  X = bag_of_words(sentence, all_words)
  X = X.reshape(1, X.shape[0])
  X = torch.from_numpy(X)

  output = model(X)
  _,predicted = torch.max(output, dim=1)
  tag = tags[predicted.item()]

  probs = torch.softmax(output, dim=1)
  prob = probs[0][predicted.item()]

if prob.item() > 0.75:
    for intent in intents['intents']:
      if tag == intent['tag']:

        if tag == 'booking':
            print(f"{bot_name}: {random.choice(intent['responses'])}")
            # Stage 1: Customer's Name
            f_name = input('\tFirst Name: ')
            l_name = input('\tLast Name: ')

            # Stage 2: Booking Dates
            arr = input('\tArrival day (DD/MM/YYYY): ')
            dep = input('\tDeparture day (DD/MM/YYYY): ')

            # Stage 3: Room and service
            room = input('\tWhich type of room are you looking for?: ')
            service = input('\tWhich service do you prefer?: ')

            # Stage 4: Confirmation and Final Rate
            c1 = Customer((f_name, l_name), (arr, dep), room, service)
            c1.ID()
            all_bookings = all_bookings.append(c1.__dict__, ignore_index=True)
            print(c1)
        elif tag == 'cancellation':
            print(f"{bot_name}: {random.choice(intent['responses'])}")
            ref_num = input('\tReference number: ')
            if ref_num in all_bookings['id'].values:
                all_bookings = all_bookings.drop(all_bookings['id'][all_bookings['id'] == ref_num].index)
                print('Your reservation has been canceled.')
            else:
                print('This reference number does not exist.')

        elif tag == 'contacts':
            contact_email = contact_dept(sentence, departments)
            if contact_email != None:
                print(f"{bot_name}: {random.choice(intent['responses'])} {contact_email}")
            else:
                print('Unfortunately this department does not exist.')
        else:
            print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        print(f'{bot_name}: I do not understand...')