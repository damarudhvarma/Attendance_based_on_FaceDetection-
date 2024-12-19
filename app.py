from flask import Flask, render_template, request
import sqlite3
from datetime import datetime

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', selected_date='', no_data=False)

@app.route('/attendance', methods=['POST'])
def attendance():
    selected_date = request.form.get('selected_date')
    selected_date_obj = datetime.strptime(selected_date, '%Y-%m-%d')
    formatted_date = selected_date_obj.strftime('%Y-%m-%d')

    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()

    # Fetch attendance data for the selected date
    cursor.execute("SELECT rollno, name, section, time FROM attendance WHERE date = ?", (formatted_date,))
    attendance_data = cursor.fetchall()

    conn.close()

    if not attendance_data:
        return render_template('index.html', selected_date=selected_date, no_data=True)
    
    return render_template('index.html', selected_date=selected_date, attendance_data=attendance_data)

@app.route('/class-attendance', methods=['POST'])
def class_attendance():
    selected_date = request.form.get('selected_date')
    selected_date_obj = datetime.strptime(selected_date, '%Y-%m-%d')
    formatted_date = selected_date_obj.strftime('%Y-%m-%d')

    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()

    # Get roll numbers marked as present for the selected date in the "attendance" table
    cursor.execute("SELECT DISTINCT rollno FROM attendance WHERE date = ?", (formatted_date,))
    present_rollnos = [row[0] for row in cursor.fetchall()]

    # Update "classRegistration" table: 1 for present roll numbers, 0 for absent
    cursor.execute("UPDATE classRegistration SET attendance = 0")  # Reset all to absent
    if present_rollnos:
        cursor.executemany(
            "UPDATE classRegistration SET attendance = 1 WHERE rollno = ?",
            [(rollno,) for rollno in present_rollnos]
        )

    # Fetch the updated "classRegistration" table
    cursor.execute("SELECT rollno, name, section, attendance FROM classRegistration")
    class_attendance_data = cursor.fetchall()

    conn.commit()
    conn.close()

    return render_template('index.html', selected_date=selected_date, class_attendance_data=class_attendance_data)

if __name__ == '__main__':
    app.run(debug=True)
