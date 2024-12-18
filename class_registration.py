import sqlite3
import tkinter as tk
from tkinter import messagebox

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

# Function to register a student
def register_student():
    rollno = entry_rollno.get().strip()
    name = entry_name.get().strip()
    
    if not rollno or not name:
        messagebox.showerror("Input Error", "Please enter both Roll No and Name.")
        return
    
    try:
        cursor.execute("INSERT INTO classRegistration (rollno, name) VALUES (?, ?)", (rollno, name))
        conn.commit()
        messagebox.showinfo("Success", f"Student {name} (Roll No: {rollno}) registered successfully!")
        entry_rollno.delete(0, tk.END)  # Clear the input fields
        entry_name.delete(0, tk.END)
    except sqlite3.IntegrityError:
        messagebox.showerror("Database Error", "Roll No already exists. Please enter a unique Roll No.")
    except Exception as e:
        messagebox.showerror("Error", f"An unexpected error occurred: {e}")

# Create the main application window
root = tk.Tk()
root.title("Class Registration")
root.geometry("800x400")  # Set the window size
root.configure(bg="#f0f0f0")  # Set a modern background color

# Create and position the heading label
heading_label = tk.Label(root, text="Class Registration", font=("Arial", 28, "bold"), bg="#f0f0f0", fg="#333")
heading_label.pack(pady=20)

# Create a frame to hold the input fields and button horizontally
frame = tk.Frame(root, bg="#f0f0f0")
frame.pack(pady=20)

# Create and position the input fields and labels
label_rollno = tk.Label(frame, text="Roll No:", font=("Arial", 14), bg="#f0f0f0", fg="#333")
label_rollno.pack(side=tk.LEFT, padx=10)
entry_rollno = tk.Entry(frame, font=("Arial", 14), width=20)  # Increased the width from 10 to 20
entry_rollno.pack(side=tk.LEFT, padx=10)

label_name = tk.Label(frame, text="Name:", font=("Arial", 14), bg="#f0f0f0", fg="#333")
label_name.pack(side=tk.LEFT, padx=10)
entry_name = tk.Entry(frame, font=("Arial", 14), width=20)
entry_name.pack(side=tk.LEFT, padx=10)

# Create and position the register button
register_button = tk.Button(frame, text="Register", font=("Arial", 14, "bold"), bg="#4CAF50", fg="white", command=register_student)
register_button.pack(side=tk.LEFT, padx=10)

# Start the Tkinter event loop
root.mainloop()

# Close the database connection when the application is closed
conn.close()
