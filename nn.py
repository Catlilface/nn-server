from http.server import BaseHTTPRequestHandler, HTTPServer
import logging
import numpy as np
import json

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def step(x):
    return 1 * (x > 0.5)


class Model:
    def __init__(self, layer_sizes):
        self.num_layers = len(layer_sizes)
        self.layer_sizes = layer_sizes

        # Инициализация весов, задаем случайные значения
        self.weights = [np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2/layer_sizes[i]) for i in range(self.num_layers-1)]

    def feedforward(self, inputs):
        """ Прямой прогон, вычисляем результат """
        self.activations = [inputs]
        for i in range(self.num_layers - 1):
            inputs = sigmoid(np.dot(inputs, self.weights[i]))
            self.activations.append(inputs)
        self.output = inputs

    def calculate(self, inputs):
        """ Вычисляем результат и выводим """
        self.feedforward(inputs)
        print(self.output)

    def backpropagation(self, inputs, targets, learning_rate):
        """ Обратный прогон, учим сеть """
        error = targets - self.output
        deltas = [error * sigmoid_derivative(self.output)]

        # Вычисляем дельты весов
        for i in range(self.num_layers-2, 0, -1):
            error = deltas[-1].dot(self.weights[i].T)
            delta = error * sigmoid_derivative(self.activations[i])
            deltas.append(delta)

        deltas = deltas[::-1]

        # Обновляем веса в соответствии с дельтами
        for i in range(self.num_layers - 1):
            self.weights[i] += self.activations[i].T.dot(deltas[i]) * learning_rate

#  --1--
# |     |
# 2     3
# |     |
#  --4--
# |     | 
# 5     6
# |     |
#  --7--

X = np.array([
    [1, 1, 1, 0, 1, 1, 1], # 0 
    [0, 0, 1, 0, 0, 1, 0], # 1 
    [1, 0, 1, 1, 1, 0, 1], # 2
    [1, 0, 1, 1, 0, 1, 1], # 3
    [0, 1, 1, 1, 0, 1, 0], # 4
    [1, 1, 0, 1, 0, 1, 1], # 5
    [1, 1, 0, 1, 1, 1, 1], # 6
    [1, 0, 1, 0, 0, 1, 0], # 7
    [1, 1, 1, 1, 1, 1, 1], # 8
    [1, 1, 1, 1, 0, 1, 1]  # 9
    ])
y = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 0
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0], # 1
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0], # 2
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0], # 3
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0], # 4
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0], # 5
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0], # 6
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0], # 7
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0], # 8
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1], # 9
    ])

global nn
nn = Model([7, 7, 6, 5, 10])

# Учим нейросеть
error = np.array([])
for i in range(10000):
    nn.feedforward(X)
    error = np.append(error, np.mean(nn.output - y))
    nn.backpropagation(X, y, 0.2)


class S(BaseHTTPRequestHandler):
    def _set_response(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

    def do_GET(self):
        logging.info("GET request,\nPath: %s\nHeaders:\n%s\n", str(self.path), str(self.headers))
        self._set_response()
        self.wfile.write("GET request for {}".format(self.path).encode('utf-8'))

    def do_POST(self):
        content_length = int(self.headers['Content-Length']) # <--- Gets the size of data
        post_data = self.rfile.read(content_length) # <--- Gets the data itself
        ar = post_data.decode('utf-8')
        ar = ar.split()
        l = [ar[4], ar[9], ar[14], ar[19], ar[24], ar[29], ar[34]]
        for i, n in enumerate(l):
            l[i] = int(n)

        self._set_response()

        nn.feedforward(l)
        output = []
        for o in nn.output:
            output.append(o.tolist())
        self.wfile.write("{}".format(json.dumps(output)).encode('utf-8'))

def run(server_class=HTTPServer, handler_class=S, port=8080):
    logging.basicConfig(level=logging.INFO)
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    logging.info('Starting httpd...\n')
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    logging.info('Stopping httpd...\n')

if __name__ == '__main__':
    from sys import argv

    if len(argv) == 2:
        run(port=int(argv[1]))
    else:
        run()