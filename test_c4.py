from keras.models import Sequential
from keras.layers import Flatten, Dense
from qlearning4k.games.connect_four import Connect
from keras.optimizers import *
from qlearning4k import Agent
from keras.models import load_model

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


model = load_model('c4.hdf5')

agent = Agent(model=model)

def opposite():
    return agent.predict(c4)

c4 = Connect(m, n, opposite=opposite)

agent.train(c4, batch_size=10, nb_epoch=50000, epsilon=.1, checkpoint=1000)
print('saving')
model.save('c4_selfplay.hdf5')

agent.play(c4, visualize=True)
