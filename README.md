# 🎓 Student-Academics-Monitoring-System
A system to manage and analyze student data using Tkinter for the UI and SQLite for storage. It supports CRUD operations, Export to CSV, and integration with machine learning algorithms like Linear Regression and K-Means. Exported data can be used in Power BI to create interactive dashboards.

---

## 🛠️ Features

- ✏️ **CRUD Operations**: Add, update, delete, and view student data.
- 🗃️ Data stored in **SQLite** (`student.db`).
- 📤 **Export to CSV** for analysis and Power BI integration.
- 📊 **Data Analysis** with machine learning algorithms: **Linear Regression**, **K-Means**, **Decision Tree**, **Pair Plot**, **Heatmap**, **Radial Plot**.
- 📈 Create **Power BI dashboards** using the exported CSV data.

---

## 🧑‍💻 How It Works

1. **Run the Application**: When the application starts, a **Tkinter UI** is displayed with options for CRUD operations.
2. **Add Data**: Enter student details (name, semester, attendance, department, GPA) and click **"Add New"** to store them in the database.
3. **Export Data**: Click the **"Export to CSV"** button to convert the database to a CSV file.
4. **Analysis**: Click **"Analysis"** to use ML algorithms like **Linear Regression**, **K-Means**, and more, with visualizations such as **Pair Plot** and **Heatmap**.
5. **Power BI**: Import the CSV file into **Power BI** to create insightful dashboards for analysis.

---

## 🏗️ Setup

### 1. Clone the Repository
bash
git clone https://github.com/your-username/student-academics-monitoring-system.git
cd student-academics-monitoring-system

---

### 2. Install Dependencies
pip install tkinter
pip install scikit-learn
pip install matplotlib
pip install pandas

---

### 3. Run the Application
python app.py

---


