import sqlite3

# Create a database connection
conn = sqlite3.connect('aptitude_test.db')
cursor = conn.cursor()

# Create a table for questions
cursor.execute('''
    CREATE TABLE IF NOT EXISTS questions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        question TEXT NOT NULL
    )
''')

# Insert sample questions
questions = [
    ("I enjoy solving complex problems.",),
    ("I prefer working in teams rather than alone.",),
    ("I find it easy to adapt to new situations.",),
    ("I like working under pressure.",),
    ("I am comfortable making decisions with limited information.",),
    ("I like to work along with my teammates",),
    ("I like to code",),
    ("I'm efficient in communication",),
    ("I love biology",),
    ("I keep myself calm in any situation",),
    # Add more questions as needed
]

cursor.executemany('INSERT INTO questions (question) VALUES (?)', questions)
conn.commit()
conn.close()
