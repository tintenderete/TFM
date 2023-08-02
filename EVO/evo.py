import numpy as np
from keras.models import Sequential
from keras.layers import Dense, GRU, Dropout, Flatten
from keras.optimizers import Adam
from loss_functions import top_is_target_31
import keras
from keras import regularizers

class GeneticAlgorithm:
    def __init__(self,
                 initial_population,
                 fitness_function,
                 selection_function,
                 crossover_function,
                 mutation_function,
                 immigration_function,
                 n_generations,
                 n_save_best_population):
        self.population = initial_population
        self.fitness_function = fitness_function
        self.selection_function = selection_function
        self.crossover_function = crossover_function
        self.mutation_function = mutation_function
        self.immigration_function = immigration_function
        self.n_generations = n_generations
        self.n_save_best_population = n_save_best_population
        self.fitness = []
    def run(self):
        for i in range(self.n_generations):
            self.fitness = self.fitness_function(self.population)
            best_population = self._get_best_population(self.population,  self.fitness)
            parents = self.selection_function(self.population, self.fitness)
            offspring = self.crossover_function(parents)
            offspring = self.mutation_function(offspring)
            self.population = np.concatenate((parents, offspring))

            immigration = self.immigration_function()
            self.population = np.concatenate((self.population, immigration))

            if(len(best_population)>0):
              self.population = np.concatenate((self.population, best_population))

        # Devuelve la mejor solución encontrada
        self.fitness = self.fitness_function(self.population)
        best_idx = np.argmax(self.fitness)
        return self.population[best_idx]

    def _get_best_population(self,population,  fitness):
      idx_best_pop = np.argsort(fitness)[::1][:self.n_save_best_population]

      return np.array(population)[idx_best_pop]
    

 # -------SIMPLE

def initial_population(population_size, chromosome_length):
    return np.random.randint(2, size=(population_size, chromosome_length))

def fitness_function(population):
    return np.sum(population, axis=1)

def selection_function(population, fitness):
    best_fitness_indices = np.argsort(fitness)[-5:]  # Get indices of top 5
    return population[best_fitness_indices]

def crossover_function(parents):
    n_offsprings = len(parents)
    offsprings = np.empty((n_offsprings, parents.shape[1]))
    crossover_point = np.uint8(parents.shape[1] / 2)

    for k in range(n_offsprings):
        # Parent selection
        parent1_idx = k%parents.shape[0]
        parent2_idx = (k+1)%parents.shape[0]
        # Performing crossover
        offsprings[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        offsprings[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    
    return offsprings

def mutation_function(offspring):
    return offspring  # no mutation is performed

def immigration_function(population_size, chromosome_length):
    return np.random.randint(2, size=(population_size, chromosome_length))


# ----------- KERAS

def generate_data(N):
    X = np.random.rand(N, 1)  # Randomly generated X values
    Y = X**3 + X**2 + X  # Corresponding Y values
    return X, Y

def initial_population_keras(population_size):
    return np.column_stack([
        np.random.randint(0, 3, size=population_size),
        np.random.randint(1, 3, size=population_size),
        np.random.randint(1, 11, size=population_size)
    ])

def create_model(genes):

    # Mapping integer genes to actual values
    activation_functions = {0: 'sigmoid', 1: 'tanh', 2: 'relu'}
    layers = {1: 1, 2: 2}
    neurons = {i: i for i in range(1, 11)}  # mapping is the same, just to keep consistency

    model = Sequential()
    activation = activation_functions[genes[0]]
    for _ in range(layers[genes[1]]):
        model.add(Dense(neurons[genes[2]], activation=activation))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer=Adam())
    return model

def fitness_function_keras(population, X, Y, epochs=5):
    fitness = np.empty(population.shape[0])
    for i in range(population.shape[0]):
        model = create_model(population[i])
        history = model.fit(X, Y, epochs=epochs, validation_split=0.2, verbose=0)
        fitness[i] = history.history['val_loss'][-1]
    return fitness

def selection_function_keras(population, fitness):
    best_fitness_indices = np.argsort(fitness)[:5]  # Get indices of 5 with lowest fitness
    return population[best_fitness_indices]

 

class I1_simple_gru:
    def __init__(self):
        self.working = True

    def initial_population(self, n_population):
        
        def get_ind():
            ind = [
                np.random.choice( [1,2,3] ) ,  # n_neurons_gru
                np.random.choice( [0, 0.1, 0.5] ),    # l1_gru
                np.random.choice( [0, 0.1, 0.5] ),    # l2_gru
                np.random.choice( [0,0.1,0.5] ) ,      # dropout
                np.random.choice( [1,2,3] ) , # n_dense_layers
                np.random.choice( [1,2,3] ) , # n_neurons_dense

            ]
            return ind

        pop = []

        for i in range(n_population):
            pop.append(get_ind())

        return np.array(pop)

    def create_model(self, genes, X_DATA, Y_DATA):
        
        n_neurons_gru = genes[0]
        l1_gru = genes[1]
        l2_gru = genes[2]
        do = genes[3]
        n_dense_layers = genes[4]
        n_neurons_dense = genes[5]

        inputs = keras.Input(shape=(X_DATA.shape[1:]))
        m = inputs

        m = GRU(units=int(n_neurons_gru), kernel_regularizer=regularizers.l1_l2(l1=l1_gru, l2=l2_gru))(m)

        m = Dropout(do)(m)

        for _ in range(int(n_dense_layers)):
            m = Dense(units=int(n_neurons_dense), activation = 'relu')(m)

        m = Flatten()(m)

        # Añadir capa Dense de salida
        out = Dense(Y_DATA.shape[1], activation='tanh')(m)

        model_GRU = keras.Model(inputs=inputs, outputs=out)

        # Compilar el modelo
        model_GRU.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss=top_is_target_31,
                metrics=[])

        return model_GRU

    def fitness_function(self, population , X_TRAIN, Y_TRAIN, X_VALID, Y_VALID, epochs = 2):
        fitness = np.empty(population.shape[0])
        for i in range(population.shape[0]):
            model = self.create_model(population[i],X_TRAIN,Y_TRAIN)
            history = model.fit(X_TRAIN, Y_TRAIN, epochs=epochs,batch_size=X_TRAIN.shape[0], validation_data = (X_VALID, Y_VALID), verbose=0)
            fitness[i] = history.history['val_loss'][-1]
        return fitness

    def selection_function(self, population, fitness, n_ind):
        best_fitness_indices = np.argsort(fitness)[:n_ind]  # Get indices of 5 with lowest fitness
        return population[best_fitness_indices]

    def crossover_function(self, parents):

        def uniform_crossover(parent1, parent2):
            child1, child2 = parent1.copy(), parent2.copy()
            for i in range(len(parent1)):
                if np.random.uniform(0, 1) < 0.5:  # Choose randomly
                    child1[i], child2[i] = child2[i], child1[i]  # Swap elements
            return child1, child2

        def easy_crossover(parents):
            parents = parents
            np.random.shuffle(parents)
            offspring = []
            for i in range(0, len(parents), 2):

                parent1 = parents[i]
                parent2 = parents[i+1]

                childs = uniform_crossover(parent1, parent2 )

                offspring.append(childs[0])
                offspring.append(childs[1])

            return offspring
        
        return easy_crossover(parents)

