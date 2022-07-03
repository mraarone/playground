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
        elif self.value > value and self.left is not None:
            return self.left.search(value)
        elif self.value < value and self.right is not None:
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
            self.items[parent], self.items[index] = (
                self.items[index],
                self.items[parent],
            )
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
        raise ValueError("Item not found")

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
        for _ in range(size):
            self.hash.append(float("inf"))

    def insert(self, value):
        for i in range(self.size):
            if self.hash[i] > value:
                self.hash[i] = value

    def delete(self, value):
        for i in range(self.size):
            if self.hash[i] == value:
                self.hash[i] = float("inf")

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
        for _ in range(size):
            self.hash.append(float("-inf"))

    def insert(self, value):
        for i in range(self.size):
            if self.hash[i] < value:
                self.hash[i] = value

    def delete(self, value):
        for i in range(self.size):
            if self.hash[i] == value:
                self.hash[i] = float("-inf")

    def search(self, value):
        for i in range(self.size):
            if self.hash[i] == value:
                return True
        return False


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
        self.model.add(
            Conv2D(
                96, (11, 11), strides=(4, 4), padding="valid", input_shape=(227, 227, 3)
            )
        )
        self.model.add(Activation("relu"))
        self.model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        self.model.add(Conv2D(256, (5, 5), padding="same"))
        self.model.add(Activation("relu"))
        self.model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        self.model.add(Conv2D(384, (3, 3), padding="same"))
        self.model.add(Activation("relu"))
        self.model.add(Conv2D(384, (3, 3), padding="same"))
        self.model.add(Activation("relu"))
        self.model.add(Conv2D(256, (3, 3), padding="same"))
        self.model.add(Activation("relu"))
        self.model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(4096))
        self.model.add(Activation("relu"))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(4096))
        self.model.add(Activation("relu"))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1000))
        self.model.add(Activation("softmax"))
        self.model.compile(
            loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"]
        )
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
        onnx.save_model(onnx_model, filename + ".onnx")
        print("Model saved to " + filename + ".onnx")


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
            self.predictions.append(
                max(set(nearest_neighbors), key=nearest_neighbors.count)
            )
        self.accuracy = sum(self.predictions == self.test_labels) / len(
            self.test_labels
        )

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
            self.test_predictions.append(
                max(set(nearest_neighbors), key=nearest_neighbors.count)
            )
        self.test_accuracy = sum(self.test_predictions == self.test_labels) / len(
            self.test_labels
        )

    # Print the accuracy of the k-nn model.
    def print_accuracy(self):
        """
        It takes in a list of training data, a list of test data, and a value for k, and then it trains a
        k-nn model on the training data and tests it on the test data
        """
        print("Accuracy: " + str(self.accuracy))
        print("Test Accuracy: " + str(self.test_accuracy))

    # Print the predictions of the k-nn model.
    def print_predictions(self):
        """
        The function takes in a dataframe, a list of features, a target column, and a list of prediction
        points, and returns a trained model, a list of predictions, and a list of test predictions.
        """
        print("Predictions: " + str(self.predictions))
        print("Test Predictions: " + str(self.test_predictions))

    # Print the accuracy and predictions of the k-nn model.
    def print_all(self):
        """
        `print_all()` prints the accuracy and predictions of the k-nn model
        """
        self.print_accuracy()
        self.print_predictions()
