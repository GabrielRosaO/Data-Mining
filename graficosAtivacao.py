import numpy as np
import matplotlib.pyplot as plt

# Definindo as funções de ativação
def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softplus(x):
    return np.log(1 + np.exp(x))

# Criando o espaço de valores para x
x = np.linspace(-10, 10, 400)

# Calculando as funções
y_relu = relu(x)
y_sigmoid = sigmoid(x)
y_softplus = softplus(x)

# Configurando o tamanho da figura para 192x192 pixels
figsize = (2.56, 2.56)  # 192 pixels equivalem a 2.56 inches (com 75 dpi)

# Função para salvar as figuras
def plot_and_save(x, y, filename, label, color):
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, y, label=label, color=color)
    ax.grid(False)
    ax.legend()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    plt.savefig(filename, dpi=75, bbox_inches='tight', pad_inches=0)
    plt.close()

# Plotando e salvando cada função de ativação
plot_and_save(x, y_relu, 'relu.png', 'ReLU', 'blue')
plot_and_save(x, y_sigmoid, 'sigmoid.png', 'Sigmoid', 'green')
plot_and_save(x, y_softplus, 'softplus.png', 'Softplus', 'red')
