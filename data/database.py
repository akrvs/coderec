import json

def read_database(db_path):
    """
    Reads the contents of a database and returns a list of words.

    Args:
        db_path: The path to the database file.

    Returns:
        A list of words extracted from the database.
    """
    words = []
    with open(db_path) as file:
        for line in file:
            line = line[:-1].replace(".", " UNK UNK UNK ").replace(",", " ").replace("!", " ").replace("  ", " ").\
                replace("?", " ").replace(":", " ").replace("-", " ")
            for word in line.split():
                words.append(word.lower())

    return words


def open_json_file(database_path):
    """
    Opens and reads a JSON file, returning the loaded data.

    Args:
        database_path (str): The path to the JSON file.

    Returns:
        dict: The loaded data from the JSON file.
    """
    with open(database_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data
