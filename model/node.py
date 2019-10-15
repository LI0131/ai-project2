class Node:

    def __init__(self, value=None):
        self.value = value if value else 0

    def set_value(self, value):
        self.value = value

    def __str__(self):
        return str(self.value)