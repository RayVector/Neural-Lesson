import numpy as np
import sys


class PartyNN(object):
    def __init__(self, learning_rate=0.1):
        self.weights_0_1 = np.random.normal(0.0, 2 ** -0.5, (2, 3))
        self.weights_1_2 = np.random.normal(0.0, 2 ** -0.5, (2, 2))
        self.weights_2_3 = np.random.normal(0.0, 1, (1, 2))
        self.sigmoid_mapper = np.vectorize(self.sigmoid)
        self.learning_rate = np.array([learning_rate])

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def predict(self, inputs):
        inputs_1 = np.dot(self.weights_0_1, inputs)
        outputs_1 = self.sigmoid_mapper(inputs_1)

        inputs_2 = np.dot(self.weights_1_2, outputs_1)
        outputs_2 = self.sigmoid_mapper(inputs_2)

        inputs_3 = np.dot(self.weights_2_3, outputs_2)
        outputs_3 = self.sigmoid_mapper(inputs_3)

        return outputs_3

    def training(self, inputs, expected_predict):
        inputs_1 = np.dot(self.weights_0_1, inputs)
        outputs_1 = self.sigmoid_mapper(inputs_1)

        inputs_2 = np.dot(self.weights_1_2, outputs_1)
        outputs_2 = self.sigmoid_mapper(inputs_2)

        inputs_3 = np.dot(self.weights_2_3, outputs_2)
        outputs_3 = self.sigmoid_mapper(inputs_3)

        actual_predict = outputs_3[0]

        error_layer_3 = np.array([actual_predict - expected_predict])
        gradient_layer_3 = actual_predict * (1 - actual_predict)
        weights_delta_layer_3 = error_layer_3 * gradient_layer_3
        self.weights_2_3 -= (np.dot(weights_delta_layer_3, outputs_2.reshape(1, len(outputs_2)))) * self.learning_rate

        error_layer_2 = weights_delta_layer_3 * self.weights_2_3
        gradient_layer_2 = outputs_2 * (1 - outputs_2)
        weights_delta_layer_2 = error_layer_2 * gradient_layer_2
        smth = np.dot(weights_delta_layer_2, outputs_1.reshape(1, len(outputs_1)))
        self.weights_1_2 -= smth * self.learning_rate

        error_layer_1 = weights_delta_layer_2 * self.weights_1_2
        gradient_layer_1 = outputs_1 * (1 - outputs_1)
        weights_delta_layer_1 = error_layer_1 * gradient_layer_1
        self.weights_0_1 -= np.dot(inputs.reshape(len(inputs), 1), weights_delta_layer_1).T * self.learning_rate


def mse(y, Y):
    return np.mean((y - Y) ** 2)


train = [
    ([0, 0, 0], 0),
    ([1, 1, 1], 1),
    ([1, 0, 1], 1),
    ([1, 1, 1], 1),
    ([1, 0, 0], 1),
]

epochs = 9000
learning_rate = 0.7

network = PartyNN(learning_rate=learning_rate)

# print('0_1: ' + str(network.weights_0_1) + '\n')
# print('1_2: ' + str(network.weights_1_2) + '\n')
# print('2_3: ' + str(network.weights_2_3) + '\n')


for e in range(epochs):
    inputs_ = []
    correct_predictions = []
    for input_stat, correct_predict in train:
        network.training(np.array(input_stat), correct_predict)
        inputs_.append(np.array(input_stat))
        correct_predictions.append(np.array(correct_predict))

    train_loss = mse(network.predict(np.array(inputs_).T), np.array(correct_predictions))
    sys.stdout.write("\rProgress: {}, Training loss: {}".format(str(100 * e / float(epochs))[:4], str(train_loss)[:5]))

print("\n")

for input_stat, correct_predict in train:
    print("For input: {} the prediction is: {}, expected: {}".format(
        str(input_stat),
        str(int(network.predict(np.array(input_stat)) > .5)),
        str(int(correct_predict == 1))
    ))

print("\n")

for input_stat, correct_predict in train:
    print("For input: {} the prediction is: {}, expected: {}".format(
        str(input_stat),
        str(network.predict(np.array(input_stat))),
        str(int(correct_predict == 1))
    ))
