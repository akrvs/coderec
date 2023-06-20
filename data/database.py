import sqlite3

def read_database():
    conn = sqlite3.connect('database.db')
    cur = conn.cursor()
    cur.execute('SELECT name FROM entry')
    rows = cur.fetchall()
    word_list = [row[0].lower() for row in rows]
    conn.close()
    return word_list