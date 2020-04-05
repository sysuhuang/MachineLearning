import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def deriv_sigmoid(x):
    fx = sigmoid(x)
    return fx * (1 - fx)


def mse_loss(y_pred, y_true):
    return ((y_pred - y_true) ** 2).mean()


class network:
  def __init__(self,learn_rate, eopchs):
    self.w1 = np.random.normal()
    self.w2 = np.random.normal()
    self.w3 = np.random.normal()
    self.w4 = np.random.normal()
    self.w5 = np.random.normal()
    self.w6 = np.random.normal()

    self.b1 = np.random.normal()
    self.b2 = np.random.normal()
    self.b3 = np.random.normal()

    self.learn_rate = learn_rate
    self.epochs = eopchs

  def feedforward(self, x):
    h1 = sigmoid(x[0] * self.w1 + x[1] * self.w2 + self.b1)
    h2 = sigmoid(x[0] * self.w3 + x[1] * self.w4 + self.b2)
    output = sigmoid(h1 * self.w4 + h2 * self.w5 + self.b3)
    return output

  def train(self, person, y_true):
    for epoch in range(self.epochs):
        for x,y in zip(person,y_true):
            sum_h1 = x[0] * self.w1 + x[1] * self.w2 + self.b1
            h1 = sigmoid(sum_h1)
            sum_h2 = x[0] * self.w3 + x[1] * self.w4 + self.b2
            h2 = sigmoid(sum_h2)
            sum_output = h1 * self.w5 + h2 * self.w6 + self.b3
            output = sigmoid(sum_output)

            d_y_y_pred = -2 * (y_true - output)

            # neuron output
            d_y_pred_w5 = h1 * deriv_sigmoid(sum_output)
            d_y_pred_w6 = h2 * deriv_sigmoid(sum_output)
            d_y_pred_b3 = deriv_sigmoid(sum_output)
            d_y_pred_h1 = self.w5 * deriv_sigmoid(sum_output)
            d_y_pred_h2 = self.w6 * deriv_sigmoid(sum_output)

            # neuron h1
            d_h1_w1 = x[0] * deriv_sigmoid(sum_h1)
            d_h1_w2 = x[1] * deriv_sigmoid(sum_h1)
            d_h1_b1 = deriv_sigmoid(sum_h1)

            # neuron h2
            d_h2_w3 = x[0] * deriv_sigmoid(sum_h2)
            d_h2_w4 = x[1] * deriv_sigmoid(sum_h2)
            d_h2_b2 = deriv_sigmoid(sum_h2)

            # Update all parameters
            # neuron output
            self.w5 -= d_y_y_pred * d_y_pred_w5 * self.learn_rate
            self.w6 -= d_y_y_pred * d_y_pred_w6 * self.learn_rate
            self.b3 -= d_y_y_pred * d_y_pred_b3 * self.learn_rate

            # neuron h1
            self.w1 -= d_y_y_pred * d_y_pred_h1 * d_h1_w1 * self.learn_rate
            self.w2 -= d_y_y_pred * d_y_pred_h1 * d_h1_w2 * self.learn_rate
            self.b1 -= d_y_y_pred * d_y_pred_h1 * d_h1_b1 * self.learn_rate

            # neuron h2
            self.w3 -= d_y_y_pred * d_y_pred_h2 * d_h2_w3 * self.learn_rate
            self.w2 -= d_y_y_pred * d_y_pred_h2 * d_h2_w4 * self.learn_rate
            self.b1 -= d_y_y_pred * d_y_pred_h2 * d_h2_b2 * self.learn_rate

        if epoch % 10 == 0:
            y_preds = np.apply_along_axis(self.feedforward, 1, data)
            loss = mse_loss(y_true, y_preds)
            print("Epoch %d loss: %.3f" % (epoch, loss))


data = np.array([
  [-2, -1],
  [25, 6],
  [17, 4],
  [-15, -6], 
])
all_y_trues = np.array([1,0,0,1])


network = network(0.1,1000)
network.train(data, all_y_trues)



