import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrowPatch

# Função de ativação (degrau)
def step_function(x):
    return np.where(x >= 0, 1, 0)

# Classe do Perceptron
class Perceptron:
    def __init__(self, input_size, learning_rate=0.1):
        self.weights = np.zeros(input_size + 1)  # +1 para o viés
        self.learning_rate = learning_rate
    
    def predict(self, x):
        summation = np.dot(x, self.weights[1:]) + self.weights[0]
        self.z.append(summation)
        return step_function(summation)
    
    def train(self, training_inputs, labels):
        self.history = []  # Para armazenar a evolução dos pesos
        self.p = [] #Para armazenar predicoes
        self.z = [] #para armazenar ponto de mudança da função degrau
        for _ in range(6):  # 10 épocas
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                self.p.append(prediction)
                self.history.append(self.weights.copy())
                self.weights[1:] += self.learning_rate * (label - prediction) * inputs
                self.weights[0] += self.learning_rate * (label - prediction)  # Atualizar o viés
        

# Dados de treinamento
training_inputs = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

labels = np.array([0, 0, 0, 1])

# Inicializando o perceptron
perceptron = Perceptron(input_size=2)
perceptron.train(training_inputs, labels)

fig, axes = plt.subplots(2, 2, figsize = (16,8))

text_ref = None

def plot_perceptron(weights, z, ax1, ax4, i, p):
    input_i = (i % 4 + 1) - 1
    ax1.clear()
    ax1.set_axis_off()
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    ajuste_x = 0.10
    font = 20

    # imput 1
    ax1.text(0.15 - ajuste_x, 0.80, str(training_inputs[input_i][0]), fontsize= font, color='g')
    ax1.text(0.15 - ajuste_x, 0.93, "x_1", fontsize= 15, color='k')

    # peso 1
    ax1.text(0.33 - ajuste_x, 0.75, f'* {weights[1]}', fontsize= font, color='y')
    ax1.text(0.36 - ajuste_x, 0.88, "w_1", fontsize= 15, color='k')
    ax1.arrow(0.20 - ajuste_x, 0.83, 0.30, -0.20, head_width=0.02, head_length=0.03)

    # ponderacao 1
    ponderacao_1 = weights[1] * training_inputs[input_i][0]
    ax1.text(0.53 - ajuste_x, 0.63, str(ponderacao_1), fontsize= font)

    # imput 2
    ax1.text(0.15 - ajuste_x, 0.20, str(training_inputs[input_i][1]), fontsize= font, color='b')
    ax1.text(0.15 - ajuste_x, 0.09, "x_2", fontsize= 15, color='k')

    # peso 2
    ax1.text(0.33 - ajuste_x, 0.25, f'* {weights[2]}', fontsize= font, color='c')
    ax1.text(0.36 - ajuste_x, 0.14, "w_2", fontsize= 15, color='k')

    ax1.arrow(0.20 - ajuste_x, 0.23, 0.30, 0.20, head_width=0.02, head_length=0.03)

    # ponderacao 2
    ponderacao_2 = weights[2] * training_inputs[input_i][1]
    ax1.text(0.53 - ajuste_x, 0.38, str(ponderacao_2), fontsize= font)

    # somatorio
    somatorio = ponderacao_1 + ponderacao_2
    ax1.text(0.60 - ajuste_x, 0.50, r'$\sum = $', fontsize= font)
    ax1.text(0.70 - ajuste_x, 0.50, f'{somatorio:.1f}', fontsize= font)

    # vies
    ax1.text(0.78 - ajuste_x, 0.58, f'+ {weights[0]:.1f}', fontsize= font, color='#ffa500')
    ax1.text(0.78 - ajuste_x, 0.71, "vies", fontsize= 15, color='k')

    ax1.arrow(0.80 - ajuste_x, 0.53, 0.07, 0, head_width=0.02, head_length=0.03)

    # somatorio + vies
    ax1.text(0.93 - ajuste_x, 0.50, f'{z:.1f}', fontsize= font)

    # y verdadeiro
    global text_ref
    if text_ref:
        text_ref.remove()
    text_ref = fig.text(0.1, 0.715, labels[input_i], fontsize = font)


    # flexa para a função de ativação
    arrow = FancyArrowPatch((0.455, 0.715) ,(0.505, 0.715), transform=fig.transFigure, 
                        arrowstyle="->", color='black', mutation_scale=20)
    fig.add_artist(arrow)

    # legenda
    point1, = ax1.plot([], [], marker="o", color='g', ls="")
    point2, = ax1.plot([], [], marker="o", color='y', ls="")
    point3, = ax1.plot([], [], marker="o", color='c', ls="")
    point4, = ax1.plot([], [], marker="o", color='b', ls="")
    point5, = ax1.plot([], [], marker="o", color='#ffa500', ls="")
    ax1.legend([point1, point2, point3, point4, point5], ['x_1' ,'w_1', 'x_2', 'w_2', 'b'], loc='lower right')

    # flexa para a atualização
    arrow2 = FancyArrowPatch((0.715, 0.50), (0.715, 0.43), transform=fig.transFigure, 
                        arrowstyle="->", color='black', mutation_scale=20)
    fig.add_artist(arrow2)

    # equação de atualização dos pesos e vies
    ax4.clear()
    ax4.set_axis_off()
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.text(0.46, 0.82, 'ŷ', fontsize = 20)

    ax4.text(0.35, 0.65 , f'$ẃ_i = w_i + 0,1(y - ŷ)x_i $', fontsize=17, color='red')
    ax4.text(0.35, 0.50 , f'$ẃ_1 = {weights[1]} + 0,1({labels[input_i]} - {p}){training_inputs[input_i][0]} $', fontsize=17, color='k')
    ax4.text(0.35, 0.35 , f'$ẃ_2 = {weights[2]} + 0,1({labels[input_i]} - {p}){training_inputs[input_i][1]} $', fontsize=17, color='k')
    ax4.text(0.35, 0.15 , f'$b´_i = b_i + 0,1(y - ŷ) $', fontsize=17, color='red')
    ax4.text(0.35, 0.0 , f'$b´ = {weights[0]:.1f} + 0,1({labels[input_i]} - {p}) $', fontsize=17, color='k')
    

def plot_decision_boundary(weights, ax, i):
    ax.clear()
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_title(f'Época {i // 4 + 1}, Iteração {i % 4 + 1}')

    # Plotar pontos de dados
    for (x, y) in zip(training_inputs, labels):
        color = 'b' if y == 1 else 'g'
        ax.scatter(x[0], x[1], c=color)

    x_vals = np.linspace(-0.1, 1.1, 200)
    y_vals = -(weights[0] + weights[1] * x_vals) / (weights[2] + 0.000001)
    ax.plot(x_vals, y_vals, 'r--')

def plot_step_function(z, p, ax, i):
    ax.clear()
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    x_vals = np.linspace(-1.1, 1.1, 400)
    y_vals = step_function(x_vals)
    ax.plot(x_vals, y_vals, label='Função Degrau')

    #plotando ponto
    color = 'g' if labels[(i % 4 + 1) - 1] - p == 0 else 'r'
    ax.scatter(z, p, c=color)



def update(i):
    ax1 = axes[0, 0]
    ax2 = axes[0, 1]
    ax3 = axes[1, 0]
    ax4 = axes[1, 1]

    # Plotar a linha de decisão
    plot_perceptron(perceptron.history[i], perceptron.z[i], ax1, ax4, i, perceptron.p[i])
    plot_step_function(perceptron.z[i], perceptron.p[i], ax2, i)
    plot_decision_boundary(perceptron.history[i], ax3, i)

# Criar a animação
ani = FuncAnimation(fig, update, frames=len(perceptron.history), interval=1000, repeat=True)

ani.save('perceptron.mp4', writer='ffmpeg', fps=1)

plt.show()
