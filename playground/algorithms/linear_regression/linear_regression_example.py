import pandas as pd
from sklearn.linear_model import LogisticRegression as logistic_regression
from sklearn.model_selection import train_test_split


class LinearRegressionExample:
    url = ""
    data = None

    def __init__(self):
        self.url = "https://raw.githubusercontent.com/noahgift/socialpowernba/\
master/data/nba_2017_players_with_salary_wiki_twitter.csv"

    def get_data(self):
        if self.url is not None:
            self.data = pd.read_csv(self.url)
        return self.data

    def process_data(self):
        # Select the columns from the dataframe. Drop columns with missing data
        self.data = self.data[["AGE", "POINTS", "SALARY_MILLIONS", "PAGEVIEWS",
            "TWITTER_FAVORITE_COUNT", "W", "TOV"]].dropna()

        # Define y (target), omega (features), and range(y_hat) (classes)
        self.target = self.data["W"].apply(lambda wins: 1 if wins > 42 else 0)
        self.features = self.data[["AGE", "POINTS", "SALARY_MILLIONS", "PAGEVIEWS",
            "TWITTER_FAVORITE_COUNT", "TOV"]]
        self.classes = ["winning", "losing"]

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.features, self.target, test_size=0.25, random_state=0)
        
        self.print_data()

    def create_model(self):
        self.model = logistic_regression(solver='lbfgs', max_iter=1000)

    def train(self):
        self.model.fit(self.x_train, self.y_train)

    def predict(self):
        print(self.model.predict(self.x_test))

    def score(self):
        print(self.model.score(self.x_test, self.y_test))

    # Print the shape and head of the data.
    def print_data(self):
        print(self.data.shape)
        print(self.data.head())
        print(self.data.describe())
        print(self.data.columns)


def main():
    lr = LogisticRegressionExample()
    lr.get_data()
    lr.process_data()
    lr.create_model()
    lr.train()
    lr.predict()
    lr.score()


if __name__ == "__main__":
    main()
