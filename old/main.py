import numpy as np
import sys

music = 1.0
rain = 1.0
girls = 0.0


random_number = np.around(np.random.uniform(low=0, high=1, size=3), decimals=2)
for z in random_number:
    print(z)


def sig(x):
    return np.around(1 / (1 + np.exp(-x)), decimals=2)


def predict(a, b, c):
    inputs = np.array([a, b, c])
    weights_hidden1 = np.around(np.random.uniform(low=0, high=1, size=3), decimals=2)
    weights_hidden2 = np.around(np.random.uniform(low=0, high=1, size=3), decimals=2)
    weights_hidden = np.array([weights_hidden1, weights_hidden2])

    weights_hidden_to_output = np.array([0.5, 0.52])

    hidden_input = np.dot(weights_hidden, inputs)
    print('hidden_input: ' + str(hidden_input))

    hidden_output = np.array([sig(x) for x in hidden_input])
    print('hidden_output: ' + str(hidden_output))

    output = np.dot(weights_hidden_to_output, hidden_output)
    print('output: ' + str(output) + '\n')

    if sig(output) < 0.5:
        return 'Yes'
    else:
        return 'No'


print('result: ' + str(predict(music, rain, girls)))
