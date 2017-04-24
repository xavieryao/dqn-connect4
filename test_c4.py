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
nb_epoch = 1000

model = Sequential()
model.add(Flatten(input_shape=(nb_frames, m, n)))
model.add(Dense(hidden_size, activation='relu'))
model.add(Dense(hidden_size, activation='relu'))
model.add(Dense(n))
model.compile(RMSprop(), "mse")




agent = Agent(model=model)

def random_play(_round=20):
    for i in range(_round):
        print("random play round {}".format(i))
        c4 = Connect(m, n)#, opposite=opposite)
        agent.train(c4, batch_size=10, nb_epoch=nb_epoch, epsilon=.1)
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
        agent.train(c4, batch_size=10, nb_epoch=nb_epoch, epsilon=.1)
        print('saving')
        model.save_weights('c4.hdf5')

def ai_play(_round=50):
    for i in range(_round):
        print("ai play round {}".format(i))
        with open('model.json', 'r') as json_file:
            stable_model = model_from_json(json_file.read())
        c4 = Connect(m, n)
        c4.opposite = play_with_ai(c4)
        agent.train(c4, batch_size=10, nb_epoch=nb_epoch, epsilon=.1)
        print('saving')
        model.save_weights('c4.hdf5')

def play_with_ai(c4):
    def func():
        import strategy
        UCT = strategy.UCTStrategy
        uct = UCT(m, n, c4.noX, c4.noY)
        uct.timeout = 10
        board = strategy.intArray(m*n)
        for i in range(m):
            for j in range(n):
                if c4.board[i][j] == 1:
                    board[i*n+j] = 2
                elif c4.board[i][j] == 2:
                    board[i*n+j] = 1
                else:
                    board[i*n+j] = 0
        top = strategy.intArray(n)
        for i in range(n):
            top[i] = int(c4.top[i])
        point = uct.getPointFor1DBoard(board=board,lastX=int(c4.last[0]),lastY=int(c4.last[1]),noX=c4.noX,noY=c4.noY,top=top,M=m,N=n)
        return point.y
    return func

def evaluate():
    c4 = Connect(m, n)
    c4.opposite = play_with_ai(c4)
    agent.play(c4, visualize=True)

if __name__ == '__main__':
    with open('model.json', 'w') as json_file:
        json_file.write(model.to_json())
    model.load_weights('c4.hdf5')
    for i in range(300):
        print("loop {}".format(i))
        ai_play(2)
        random_play(2)
        # self_play(1)
        evaluate()
