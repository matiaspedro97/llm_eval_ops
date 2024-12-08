import json
from weave import Dataset

class DatasetLoader:
    def __init__(self, data_input: str, name: str = 'my_dataset'):
        """
        Initialize the DatasetLoader with either a dictionary or a JSON file path.
        
        :param data_input: A dictionary or the path to a JSON file.
        """
        self.data_input = data_input
        self.data = self._load_data()

        self.name = name

    def _load_data(self):
        """
        Load the data from the provided input (dictionary or file path).
        
        :return: The data as a list of dictionaries.
        """
        if isinstance(self.data_input, dict):
            # If it's already a dictionary, just use it
            return self.data_input
        elif isinstance(self.data_input, str):
            # If it's a string, assume it's a file path and load the JSON from file
            with open(self.data_input, 'r') as file:
                return json.load(file)
        else:
            raise ValueError("Input must be a dictionary or a path to a JSON file.")

    def to_weave_dataset(self):
        """
        Transform the loaded data into a weave.Dataset object.
        
        :param name: The name of the dataset (default is "my_dataset").
        :return: A weave.Dataset object.
        """
        if not isinstance(self.data, list):
            raise ValueError("Data should be a list of dictionaries.")
        
        # Ensure that each item in the list has the required fields "question" and "rubric"
        rows = []
        for item in self.data:
            rows.append(item)

        return Dataset(
            name=self.name,
            rows=rows
        )