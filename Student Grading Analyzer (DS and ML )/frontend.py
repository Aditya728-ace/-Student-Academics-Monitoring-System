from tkinter import *
from tkinter import ttk  
import backend
import ml  

import subprocess  

def get_selected_row(event):
    global selected_tuple
    if lb1.curselection() != ():
        index = lb1.curselection()[0]
        selected_tuple = lb1.get(index)
        clear_entries()
        e1.insert(END, selected_tuple[1])
        e2.set(selected_tuple[2]) 
        e3.insert(END, selected_tuple[3])
        e4.insert(END, selected_tuple[4])
        e5.insert(END, selected_tuple[5])  

def view_command():
    lb1.delete(0, END)
    for row in backend.view():
        lb1.insert(END, row)

def search_command():
    lb1.delete(0, END)
    for row in backend.search(Name.get(), Department.get(), Semester.get(), GPA.get(), Attendance.get()):
        lb1.insert(END, row)
    clear_entries()

def add_command():
    backend.insert(Name.get(), Department.get(), Semester.get(), GPA.get(), Attendance.get())
    clear_entries()
    view_command()

def update_command():
    backend.update(selected_tuple[0], Name.get(), Department.get(), Semester.get(), GPA.get(), Attendance.get())
    clear_entries()
    view_command()

def delete_command():
    index = lb1.curselection()[0]
    selected_tuple = lb1.get(index)
    backend.delete(selected_tuple[0])
    clear_entries()
    view_command()

def delete_data_command():
    backend.delete_data()
    view_command()

def clear_entries():
    e1.delete(0, END)
    e2.set('') 
    e3.delete(0, END)
    e4.delete(0, END)
    e5.delete(0, END)  

def clear_command():
    lb1.delete(0, END)
    clear_entries()

def export_to_csv_command():
    backend.db_to_csv("students_data.csv")
    clear_entries()

def start_analysis():
    subprocess.Popen(['python', 'ml.py']) 
    

wind = Tk()

Name = StringVar()
Department = StringVar()
Semester = StringVar()
GPA = StringVar()
Attendance = StringVar() 

l0 = Label(wind, text="Students", width="10", fg="blue")
l0.config(font=("Courier", 15))

l00 = Label(wind, text="Database", width="10", fg="blue")
l00.config(font=("Courier", 15))

l1 = Label(wind, text="First Name", width="10")
l2 = Label(wind, text="Department", width="10")
l3 = Label(wind, text="Term", width="10")
l4 = Label(wind, text="GPA", width="10")
l5 = Label(wind, text="Attendance", width="10")  

e1 = Entry(wind, textvariable=Name)
e2 = ttk.Combobox(wind, textvariable=Department)  
e3 = ttk.Combobox(wind, textvariable=Semester)
e4 = Entry(wind, textvariable=GPA)
e5 = Entry(wind, textvariable=Attendance)  

#COMBOX USE KIYA HAI FOR MULTIPLE PREDEFINED VALUES

departments = ["Computer Science", "Electrical Engineering", "Mechanical Engineering", "Civil Engineering", "Information Technology"]
e2['values'] = departments  
e2.set('Select Department')  


semester = ["1", "2", "3", "4", "5", "6", "7","8"]
e3['values'] = semester 
e3.set('Select Semester')  

# Buttons
b1 = Button(wind, text="View all", width="15", command=view_command)
b2 = Button(wind, text="Search", width="15", command=search_command)
b3 = Button(wind, text="Add New", width="15", command=add_command)
b4 = Button(wind, text="Update", width="15", command=update_command)
b5 = Button(wind, text="Delete", width="15", command=delete_command)
b6 = Button(wind, text="Clear", width="15", command=clear_command)
b7 = Button(wind, text="Delete all Students", width="15", command=delete_data_command)
b8 = Button(wind, text="Export to CSV", width="15", command=export_to_csv_command)  
b9 = Button(wind, text="Exit", width="15", command=wind.destroy)

b10 = Button(wind, text="Analysis", width="15", command=start_analysis)

lb1 = Listbox(wind, height=6, width=35)
lb1.bind('<<ListboxSelect>>', get_selected_row)

sc = Scrollbar(wind)


l0.grid(row=0, column=1)
l00.grid(row=0, column=2)
l1.grid(row=1, column=0)
l2.grid(row=1, column=2)
l3.grid(row=2, column=0)
l4.grid(row=2, column=2)
l5.grid(row=3, column=0)  

e1.grid(row=1, column=1)
e2.grid(row=1, column=3)
e3.grid(row=2, column=1)
e4.grid(row=2, column=3)
e5.grid(row=3, column=1)  
b1.grid(row=3, column=3)
b2.grid(row=4, column=3)
b3.grid(row=5, column=3)
b4.grid(row=6, column=3)
b5.grid(row=7, column=3)
b6.grid(row=8, column=3)
b7.grid(row=9, column=3)
b8.grid(row=10, column=3)  
b9.grid(row=11, column=3)

b10.grid(row=12, column=3)  

lb1.grid(row=4, column=0, rowspan=8, columnspan=2)
sc.grid(row=4, column=2, rowspan=8)

lb1.configure(yscrollcommand=sc.set)
sc.configure(command=lb1.yview)

wind.mainloop()
