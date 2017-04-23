from keras.models import Sequential
from keras.layers import Flatten, Dense
from qlearning4k.games.connect_four import Connect
from keras.optimizers import *
from qlearning4k import Agent
from keras.models import load_model
from keras.models import model_from_json
import numpy as np

m = 10
n = 10
hidden_size = 200
nb_frames = 1

model = Sequential()
model.add(Flatten(input_shape=(nb_frames, m, n)))
model.add(Dense(hidden_size, activation='relu'))
model.add(Dense(hidden_size, activation='relu'))
model.add(Dense(n))
model.compile(RMSprop(), "mse")


with open('model.json', 'w') as json_file:
    json_file.write(model.to_json())

model.load_weights('c4.hdf5')

agent = Agent(model=model)

def random_play(_round=20):
    for i in range(_round):
        print("random play round {}".format(i))
        c4 = Connect(m, n)#, opposite=opposite)
        agent.train(c4, batch_size=10, nb_epoch=5000, epsilon=.1)
        print('saving')
        model.save_weights('c4.hdf5')

def self_play(_round=50):
    for i in range(_round):
        print("self play round {}".format(i))
        with open('model.json', 'r') as json_file:
            stable_model = model_from_json(json_file.read())
        stable_model.load_weights('c4.hdf5')
        stable_agent = Agent(model=stable_model)
        mirror_game = Connect(m, n)
        def opposite():
            mirror_game.board = np.array(c4.get_state())
            for i in range(m):
                for j in range(n):
                    if mirror_game.board[i][j] == 1:
                        mirror_game.board[i][j] = 2
                    elif mirror_game.board[i][j] == 2:
                        mirror_game.board[i][j] = 1
            return stable_agent.predict(mirror_game)

        c4 = Connect(m, n, opposite=opposite)
        agent.train(c4, batch_size=10, nb_epoch=5000, epsilon=.1)
        print('saving')
        model.save_weights('c4.hdf5')

def evaluate():
    c4 = Connect(m, n)
    agent.play(c4, visualize=True)

if __name__ == '__main__':
    for i in range(30):
        print("loop {}".format(i))
        random_play(1)
        self_play(3)
        evaluate()
