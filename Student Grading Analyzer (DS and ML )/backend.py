import sqlite3
import os
import csv

def connect():
    conn = sqlite3.connect("Students.db")
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS data2 (
        id INTEGER PRIMARY KEY, 
        Name TEXT, 
        Department TEXT, 
        Semester INTEGER, 
        GPA REAL,
        Attendance INTEGER  -- Added Attendance column
    )
    """)
    conn.commit()
    conn.close()

def insert(Name, Department, Semester, GPA, Attendance):
    try:
        with sqlite3.connect("Students.db") as conn:
            cur = conn.cursor()
            cur.execute("INSERT INTO data2 (Name, Department, Semester, GPA, Attendance) VALUES (?,?,?,?,?)", 
                        (Name, Department, Semester, GPA, Attendance))
            conn.commit()
    except sqlite3.Error as e:
        print(f"Error while inserting record: {e}")

def view():
    conn = sqlite3.connect("Students.db")
    cur = conn.cursor()
    cur.execute("SELECT * FROM data2")
    rows = cur.fetchall()
    conn.close()
    return rows

def search(Name="", Department="", Semester="", GPA="", Attendance=""):
    conn = sqlite3.connect("Students.db")
    cur = conn.cursor()

    query = "SELECT * FROM data2 WHERE 1=1" 
    params = []

    if Name:
        query += " AND Name=?"
        params.append(Name)
    if Department:
        query += " AND Department=?"
        params.append(Department)
    if Semester:
        query += " AND Semester=?"
        params.append(Semester)
    if GPA:
        query += " AND GPA=?"
        params.append(GPA)
    if Attendance:
        query += " AND Attendance=?"
        params.append(Attendance)

    cur.execute(query, tuple(params))
    rows = cur.fetchall()
    conn.close()
    return rows

def delete(id):
    try:
        conn = sqlite3.connect("Students.db")
        cur = conn.cursor()
        cur.execute("DELETE FROM data2 WHERE id=?", (id,))
        conn.commit()
        conn.close()
    except sqlite3.Error as e:
        print(f"Error while deleting record: {e}")

def update(id, Name, Department, Semester, GPA, Attendance):
    try:
        conn = sqlite3.connect("Students.db")
        cur = conn.cursor()
        cur.execute("UPDATE data2 SET Name=?, Department=?, Semester=?, GPA=?, Attendance=? WHERE id=?", 
                    (Name, Department, Semester, GPA, Attendance, id))
        conn.commit()
        conn.close()
    except sqlite3.Error as e:
        print(f"Error while updating record: {e}")

def delete_data():
    try:
        if os.path.exists("Students.db"):
            os.remove("Students.db")
        connect()
    except Exception as e:
        print(f"Error while deleting database: {e}")

def db_to_csv(csv_file):
    try:
        conn = sqlite3.connect("Students.db")
        cur = conn.cursor()
        cur.execute("SELECT * FROM data2")
        rows = cur.fetchall()

        with open(csv_file, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([description[0] for description in cur.description])  
            writer.writerows(rows)  
        conn.close()
        print(f"Data successfully exported to {csv_file}")
    except sqlite3.Error as e:
        print(f"Error while exporting to CSV: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

connect()  
