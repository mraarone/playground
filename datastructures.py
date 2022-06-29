# Lists and Arrays

# This class provides a data structure for a binary tree.
class BinaryTrees:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

    def insert_left(self, value):
        self.left = BinaryTrees(value)
        return self.left

    def insert_right(self, value):
        self.right = BinaryTrees(value)
        return self.right

    def search(self, value):
        if self.value == value:
            return True
        elif self.value > value:
            if self.left is not None:
                return self.left.search(value)
        elif self.value < value:
            if self.right is not None:
                return self.right.search(value)
        return False

    def print_tree(self):
        print(self.value)
        if self.left is not None:
            self.left.print_tree()
        if self.right is not None:
            self.right.print_tree()

# This class named Queue provides a data structure for a Queue.
class Queue:
    def __init__(self):
        self.items = []

    def is_empty(self):
        return self.items == []

    def enqueue(self, item):
        self.items.insert(0, item)

    def dequeue(self):
        return self.items.pop()

    def size(self):
        return len(self.items)

# This class named BinaryHeap provides a data structure for a Binary Heap.
class BinaryHeap:
    def __init__(self):
        self.items = []

    def is_empty(self):
        return self.items == []

    def insert(self, item):
        self.items.append(item)
        self.bubble_up(len(self.items) - 1)

    def size(self):
        return len(self.items)

    def bubble_up(self, index):
        parent = (index - 1) // 2
        if parent >= 0 and self.items[parent] < self.items[index]:
            self.items[parent], self.items[index] = self.items[index], self.items[parent]
            self.bubble_up(parent)

    def bubble_down(self, index):
        left = 2 * index + 1
        right = 2 * index + 2
        if left < len(self.items) and self.items[index] < self.items[left]:
            self.items[index], self.items[left] = self.items[left], self.items[index]
            self.bubble_down(left)
        if right < len(self.items) and self.items[index] < self.items[right]:
            self.items[index], self.items[right] = self.items[right], self.items[index]
            self.bubble_down(right)

    def delete(self, item):
        for i in range(len(self.items)):
            if self.items[i] == item:
                self.items[i] = self.items[-1]
                self.items.pop()
                self.bubble_down(i)
                return
        raise ValueError('Item not found')

    def delete_min(self):
        self.items[0] = self.items[-1]
        self.items.pop()
        self.bubble_down(0)

    def peek(self):
        return self.items[0]

# This class named Stack provides a data structure for a Stack.
class Stack:
    def __init__(self):
        self.items = []

    def is_empty(self):
        return self.items == []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        return self.items.pop()

    def peek(self):
        return self.items[-1]

    def size(self):
        return len(self.items)

# This class named MinHash provides a data structure for a MinHash.
class MinHash:
    def __init__(self, size):
        self.size = size
        self.hash = []
        for i in range(size):
            self.hash.append(float('inf'))

    def insert(self, value):
        for i in range(self.size):
            if self.hash[i] > value:
                self.hash[i] = value

    def delete(self, value):
        for i in range(self.size):
            if self.hash[i] == value:
                self.hash[i] = float('inf')

    def search(self, value):
        for i in range(self.size):
            if self.hash[i] == value:
                return True
        return False

# This class named MaxHash provides a data structure for a MaxHash.
class MaxHash:
    def __init__(self, size):
        self.size = size
        self.hash = []
        for i in range(size):
            self.hash.append(float('-inf'))

    def insert(self, value):
        for i in range(self.size):
            if self.hash[i] < value:
                self.hash[i] = value

    def delete(self, value):
        for i in range(self.size):
            if self.hash[i] == value:
                self.hash[i] = float('-inf')

    def search(self, value):
        for i in range(self.size):
            if self.hash[i] == value:
                return True
        return False

# This class named DAG provides a data structure for a Directed Acyclic Graph.
class DAG:
    def __init__(self):
        self.nodes = []

    def add_node(self, node):
        self.nodes.append(node)

    def add_edge(self, node1, node2):
        for i in range(len(self.nodes)):
            if self.nodes[i] == node1:
                self.nodes[i].add_child(node2)
                return
        raise ValueError('Node not found')

    def print_graph(self):
        for node in self.nodes:
            node.print_node()


import pandas as pd
import numpy as np

# This class implements pandas to download wikipedia data from the internet.
class Wikipedia:
    def __init__(self):
        self.df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/wikipedia/en-wiki-big.csv', sep='\t')

    def get_df(self):
        return self.df

    def get_data(self):
        return self.df.values

    def get_columns(self):
        return self.df.columns

    def get_rows(self):
        return self.df.shape[0]

    def get_row(self, row):
        return self.df.iloc[row]

    def get_row_data(self, row):
        return self.df.iloc[row].values

    def get_column(self, column):
        return self.df[column]

    def get_column_data(self, column):
        return self.df[column].values

# Create a graph class.
class Graph:
    def __init__(self):
        self.nodes = []

    def add_node(self, node):
        self.nodes.append(node)

    def add_edge(self, node1, node2):
        for i in range(len(self.nodes)):
            if self.nodes[i] == node1:
                self.nodes[i].add_child(node2)
                return
        raise ValueError('Node not found')

    def print_graph(self):
        for node in self.nodes:
            node.print_node()

# Create a Lightmap class.
class Lightmap:
    def __init__(self, size):
        self.size = size
        self.lightmap = []
        for i in range(size):
            self.lightmap.append(0)

    def insert(self, value):
        for i in range(self.size):
            if self.lightmap[i] == 0:
                self.lightmap[i] = value
                return
        raise ValueError('Lightmap is full')

    def delete(self, value):
        for i in range(self.size):
            if self.lightmap[i] == value:
                self.lightmap[i] = 0
                return
        raise ValueError('Value not found')

    def search(self, value):
        for i in range(self.size):
            if self.lightmap[i] == value:
                return True
        return False

# Create a RadixTree class.
class RadixTree:
    def __init__(self):
        self.root = Node()

    def insert(self, word):
        current = self.root
        for i in range(len(word)):
            if current.children[word[i]] == None:
                current.children[word[i]] = Node()
            current = current.children[word[i]]
        current.is_word = True

    def search(self, word):
        current = self.root
        for i in range(len(word)):
            if current.children[word[i]] == None:
                return False
            current = current.children[word[i]]
        return current.is_word

    def print_tree(self):
        self.root.print_node()

# Create a DoublyLinkedList class.
class DoublyLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None

    def add_to_head(self, value):
        new_node = Node(value)
        if self.head == None:
            self.head = new_node
            self.tail = new_node
        else:
            new_node.next = self.head
            self.head.prev = new_node
            self.head = new_node

    def add_to_tail(self, value):
        new_node = Node(value)
        if self.tail == None:
            self.head = new_node
            self.tail = new_node
        else:
            new_node.prev = self.tail
            self.tail.next = new_node
            self.tail = new_node

    def remove_from_head(self):
        if self.head == None:
            return None
        else:
            value = self.head.value
            self.head = self.head.next
            if self.head != None:
                self.head.prev = None
            return value

    def remove_from_tail(self):
        if self.tail == None:
            return None
        else:
            value = self.tail.value
            self.tail = self.tail.prev
            if self.tail != None:
                self.tail.next = None
            return value

    def print_list(self):
        current = self.head
        while current != None:
            print(current.value)
            current = current.next

# Create a Set class.
class Set:
    def __init__(self):
        self.set = []

    def add(self, value):
        if value not in self.set:
            self.set.append(value)

    def remove(self, value):
        if value in self.set:
            self.set.remove(value)

    def search(self, value):
        if value in self.set:
            return True
        return False

    def print_set(self):
        for i in range(len(self.set)):
            print(self.set[i])

# Create a keras model based on the AlexNet architecture.
class AlexNet:
    def __init__(self):
        self.model = Sequential()
        self.model.add(Conv2D(96, (11, 11), strides=(4, 4), padding='valid', input_shape=(227, 227, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        self.model.add(Conv2D(256, (5, 5), padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        self.model.add(Conv2D(384, (3, 3), padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(384, (3, 3), padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(256, (3, 3), padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(4096))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(4096))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1000))
        self.model.add(Activation('softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
        self.model.summary()

    def train(self, X, Y, batch_size, epochs):
        self.model.fit(X, Y, batch_size=batch_size, epochs=epochs)
    
    def test(self, X, Y):
        self.model.evaluate(X, Y)
    
    def predict(self, X):
        return self.model.predict(X)
    
    # Convert model to ONNX and save.
    def save(self, filename):
        self.model.save(filename)
        onnx_model = keras2onnx.convert_keras(self.model, self.model.name)
        onnx.save_model(onnx_model, filename + '.onnx')
        print('Model saved to ' + filename + '.onnx')

# Create a class the creates a 16 by 16 grid, and initializes all values to 0, and provides a function to draw a snake of 1's within the grid, implementing the snake game.
class Grid:
    def __init__(self):
        self.grid = [[0 for i in range(16)] for j in range(16)]
        self.snake = []
        self.snake_direction = 'right'
        self.snake_length = 3
        self.snake_head = (0, 0)
        self.snake_tail = (0, 0)
        self.food = (0, 0)
        self.food_present = False
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.game_lost = False
        self.game_paused = False
        self.game_started = False
        self.game_reset = False
        self.game_quit = False
        self.game_exit = False
        self.game_restart = False
        self.game_resume = False
        self.game_pause = False
        self.game_start = False
        self.game_quit = False
        self.game_reset = False
        self.game_exit = False
        self.game_restart = False
        self.game_resume = False
        self.game_pause = False
        self.game_start = False
        self.game_quit = False
        self.game_reset = False
        self.game_exit = False
        self.game_restart = False
        self.game_resume = False
        self.game_pause = False
        self.game_start = False
        self.game_quit = False
        self.game_reset = False
        self.game_exit = False
        self.game_restart = False
        self.game_resume = False
        self.game_pause = False
        self.game_start = False
        self.game_quit = False
        self.game_reset = False
        self.game_exit = False
        self.game_restart = False
        self.game_resume = False
        self.game_pause = False
        self.game_start = False
        self.game_quit = False
    
    # Draw a snake of 1's within the grid that can be oriented along the x axis, y axis, or diagonally.
    def draw_snake(self):
        if self.snake_direction == 'right':
            for i in range(self.snake_length):
                self.grid[self.snake_head[0]][self.snake_head[1] + i] = 1
        elif self.snake_direction == 'left':
            for i in range(self.snake_length):
                self.grid[self.snake_head[0]][self.snake_head[1] - i] = 1
        elif self.snake_direction == 'up':
            for i in range(self.snake_length):
                self.grid[self.snake_head[0] - i][self.snake_head[1]] = 1
        elif self.snake_direction == 'down':
            for i in range(self.snake_length):
                self.grid[self.snake_head[0] + i][self.snake_head[1]] = 1
        elif self.snake_direction == 'upright':
            for i in range(self.snake_length):
                self.grid[self.snake_head[0] - i][self.snake_head[1] + i] = 1
        elif self.snake_direction == 'upleft':
            for i in range(self.snake_length):
                self.grid[self.snake_head[0] - i][self.snake_head[1] - i] = 1
        elif self.snake_direction == 'downright':
            for i in range(self.snake_length):
                self.grid[self.snake_head[0] + i][self.snake_head[1] + i] = 1
        elif self.snake_direction == 'downleft':
            for i in range(self.snake_length):
                self.grid[self.snake_head[0] + i][self.snake_head[1] - i] = 1
        else:
            print('Error: invalid snake direction.')

    # Draw a food item within the grid.
    def draw_food(self):
        self.food = (random.randint(0, 15), random.randint(0, 15))
        self.grid[self.food[0]][self.food[1]] = 2
        self.food_present = True
    
    # Check if the snake has collided with the wall or itself.
    def check_collision(self):
        if self.snake_head[0] < 0 or self.snake_head[0] > 15 or self.snake_head[1] < 0 or self.snake_head[1] > 15:
            self.game_over = True
        for i in range(self.snake_length):
            if self.snake_head[0] == self.snake[i][0] and self.snake_head[1] == self.snake[i][1]:
                self.game_over = True
        if self.snake_head[0] == self.food[0] and self.snake_head[1] == self.food[1]:
            self.food_present = False
            self.score += 1
            self.snake_length += 1
            self.draw_food()
    
    # Move the snake forward one space.
    def move_snake(self):
        self.snake_tail = self.snake_head
        self.snake_head = (self.snake_head[0] + self.snake_direction[0], self.snake_head[1] + self.snake_direction[1])
        self.snake.append(self.snake_head)
        self.grid[self.snake_tail[0]][self.snake_tail[1]] = 0
        self.grid[self.snake_head[0]][self.snake_head[1]] = 1
        self.check_collision()

    
    
# A class that uses pytorch to create a model based on AlexNex architecture
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.5)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.relu(
    
    def train(self, x, y):
        self.optimizer.zero_grad()
        output = self.forward(x)
        loss = self.criterion(output, y)
        loss.backward()
        self.optimizer.step()
        return loss
    
    def test(self, x, y):
        output = self.forward(x)
        loss = self.criterion(output, y)
        return loss
    
    def predict(self, x):
        output = self.forward(x)
        return self.sigmoid(output)

# A k-nn class that uses the k-nn algorithm to predict the class of a given data point.

# It takes in a k value, a training set, and a test set, and then it finds the k nearest neighbors of
# each test data point and predicts the label of the test data point based on the labels of the k
# nearest neighbors.
class KNN:
    def __init__(self, k):
        self.k = k
        self.data = []
        self.labels = []
        self.test_data = []
        self.test_labels = []
        self.predictions = []
        self.test_predictions = []
        self.accuracy = 0
        self.test_accuracy = 0
    
    # Add a data point to the training set.
    def add_data(self, data, label):
        self.data.append(data)
        self.labels.append(label)
    
    # Add a data point to the test set.
        """
        It adds a data point to the test set
        
        :param data: The data point to add to the test set
        :param label: The label of the data point
        """
    def add_test_data(self, data, label):
        self.test_data.append(data)
        self.test_labels.append(label)
    
    # Train the k-nn model.
    def train(self):
        """
        For each test data point, find the distance between it and every training data point, sort the
        distances, and then find the labels of the k nearest neighbors.
        """
        self.predictions = []
        for i in range(len(self.test_data)):
            distances = []
            for j in range(len(self.data)):
                distances.append(np.linalg.norm(self.test_data[i] - self.data[j]))
            distances = np.array(distances)
            indices = np.argsort(distances)
            nearest_neighbors = []
            for k in range(self.k):
                nearest_neighbors.append(self.labels[indices[k]])
            self.predictions.append(max(set(nearest_neighbors), key=nearest_neighbors.count))
        self.accuracy = sum(self.predictions == self.test_labels) / len(self.test_labels)
    
    # Test the k-nn model.
    def test(self):
        """
        For each test data point, find the distance between it and every training data point, sort the
        distances, and then find the most common label among the k nearest neighbors.
        """
        self.test_predictions = []
        for i in range(len(self.test_data)):
            distances = []
            for j in range(len(self.data)):
                distances.append(np.linalg.norm(self.test_data[i] - self.data[j]))
            distances = np.array(distances)
            indices = np.argsort(distances)
            nearest_neighbors = []
            for k in range(self.k):
                nearest_neighbors.append(self.labels[indices[k]])
            self.test_predictions.append(max(set(nearest_neighbors), key=nearest_neighbors.count))
        self.test_accuracy = sum(self.test_predictions == self.test_labels) / len(self.test_labels)
    
    # Print the accuracy of the k-nn model.
        """
        It takes in a list of training data, a list of test data, and a value for k, and then it trains a
        k-nn model on the training data and tests it on the test data
        """
    def print_accuracy(self):
        print("Accuracy: " + str(self.accuracy))
        print("Test Accuracy: " + str(self.test_accuracy))
    
    # Print the predictions of the k-nn model.
        """
        The function takes in a dataframe, a list of features, a target column, and a list of prediction
        points, and returns a trained model, a list of predictions, and a list of test predictions.
        """
    def print_predictions(self):
        print("Predictions: " + str(self.predictions))
        print("Test Predictions: " + str(self.test_predictions))
    
    # Print the accuracy and predictions of the k-nn model.
        """
        `print_all()` prints the accuracy and predictions of the k-nn model
        """
    def print_all(self):
        self.print_accuracy()
        self.print_predictions()

