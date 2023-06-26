import sqlite3

def read_database(db_path):
    """
    Reads the contents of a database and returns a list of words.

    Args:
        db_path: The path to the database file.

    Returns:
        A list of words extracted from the database.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute('SELECT name FROM entry')
    rows = cur.fetchall()
    word_list = [row[0].lower() for row in rows]
    conn.close()
    return word_list