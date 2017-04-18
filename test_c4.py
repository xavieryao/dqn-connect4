from keras.models import Sequential
from keras.layers import Flatten, Dense
from qlearning4k.games.connect_four import Connect
from keras.optimizers import *
from qlearning4k import Agent
from keras.models import load_model
from keras.models import model_from_json

m = 7
n = 8
hidden_size = 100
nb_frames = 1

model = Sequential()
model.add(Flatten(input_shape=(nb_frames, m, n)))
model.add(Dense(hidden_size, activation='relu'))
model.add(Dense(hidden_size, activation='relu'))
model.add(Dense(8))
model.compile(sgd(lr=.2), "mse")


model.load_weights('c4.hdf5')
with open('model.json', 'w') as json_file:
    json_file.write(model.to_json())
agent = Agent(model=model)
'''
for i in range(10):
    print("self play round {}".format(i))
    with open('model.json', 'r') as json_file:
        stable_model = model_from_json(json_file.read())
    stable_model.load_weights('c4.hdf5')
    #stable_agent = Agent(model=stable_model)
    #def opposite():
    #    return stable_agent.predict(c4)

    c4 = Connect(m, n)#, opposite=opposite)
    agent.train(c4, batch_size=10, nb_epoch=1000, epsilon=.1, checkpoint=1000)
    print('saving')
    model.save_weights('c4.hdf5')
'''
c4 = Connect(m, n)
agent.play(c4, visualize=True)
