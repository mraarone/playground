from copy import deepcopy


class Singleton:
    """
    Singleton class
    """

    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, "instance"):
            cls.instance = super(Singleton, cls).__new__(cls, *args, **kwargs)
        return cls.instance

    @staticmethod
    def static_method():
        "Use @classmethod to if access to class level variables is not needed."

    @classmethod
    def class_method(cls):
        "Use @staticmethod to access class level variables."
        print(cls.instance)


def main():
    singleton = Singleton()
    singleton.class_method()
    singleton.static_method()

    singleton_copy = deepcopy(singleton)
    print(
        "Singleton copy and Singleton are the same? ",
        id(singleton_copy) == id(singleton),
    )


if __name__ == "__main__":
    main()
