import sqlite3
import json

def read_database(db_path):
    """
    Reads the contents of a database and returns a list of words.

    Args:
        db_path: The path to the database file.

    Returns:
        A list of words extracted from the database.
    """
    '''conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute('SELECT name FROM entry')
    rows = cur.fetchall()
    word_list = [row[0].lower() for row in rows]
    conn.close()
    return word_list'''
    words = []
    with open(db_path) as file:
        for line in file:
            line = line[:-1].replace(".", " UNK UNK UNK ").replace(",", " ").replace("!", " ").replace("  ", " ").\
                replace("?", " ").replace(":", " ").replace("-", " ")
            for word in line.split():
                words.append(word.lower())

    return words


def open_json_file(database_path):
    with open(database_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data