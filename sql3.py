import sqlite3

# Connect to the SQLite database
conn = sqlite3.connect('attendance.db')
cursor = conn.cursor()

# Execute a query
cursor.execute("SELECT * FROM attendance")

# Fetch all rows from the query result
rows = cursor.fetchall()

# Print the rows
for row in rows:
    print(row)

# Close the connection
conn.close()
