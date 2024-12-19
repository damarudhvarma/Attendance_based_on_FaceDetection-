import sqlite3
import csv
import os

# Path to the CSV file
csv_path = os.path.join(os.path.dirname(__file__), 'data', 'features_all.csv')

# Create or connect to the database
conn = sqlite3.connect('attendance.db')
cursor = conn.cursor()

# Create the 'classRegistration' table if it doesn't exist
cursor.execute('''
CREATE TABLE IF NOT EXISTS classRegistration (
    rollno VARCHAR PRIMARY KEY,
    name TEXT NOT NULL,
    section TEXT DEFAULT 'CSE-B',
    attendance BOOLEAN DEFAULT 0
)
''')
conn.commit()

# Function to register students from the CSV file
def register_students_from_csv(file_path):
    try:
        with open(file_path, 'r') as file:
            csv_reader = csv.reader(file)
            
            for row_num, row in enumerate(csv_reader, start=2):  # Start counting from 2 (assuming row 1 is the header)
                if len(row) < 2:
                    print(f"Row {row_num} is missing required data, skipping...")
                    continue

                rollno, name = row[0].strip(), row[1].strip()
                
                if not rollno or not name:
                    print(f"Row {row_num} has empty rollno or name, skipping...")
                    continue

                try:
                    cursor.execute("INSERT INTO classRegistration (rollno, name) VALUES (?, ?)", (rollno, name))
                    print(f"Successfully registered student {name} (Roll No: {rollno})")
                except sqlite3.IntegrityError:
                    print(f"Roll No {rollno} already exists, skipping...")
                except Exception as e:
                    print(f"Error inserting row {row_num}: {e}")

        conn.commit()
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Register students from the CSV file
register_students_from_csv(csv_path)

# Close the database connection
conn.close()
