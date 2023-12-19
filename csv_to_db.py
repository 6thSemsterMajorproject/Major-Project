# import sqlite3
# import csv

# # Step 1: Choose SQLite and create a connection
# conn = sqlite3.connect('C:\\AIhealthpro\\instance\\AIhealthpro.db')
# cursor = conn.cursor()

# # Step 2: Create a table (You can define your table creation query here)
# # For example:
# # cursor.execute('''CREATE TABLE IF NOT EXISTS diabete (
# #                     col1 TEXT,
# #                     col2 TEXT,
# #                     col3 TEXT,
# #                     ...
# #                 )''')

# # Step 3: Read the CSV file
# with open("heart.csv", 'r') as file:
#     # Create a CSV reader
#     csv_reader = csv.reader(file)
#     # Skip the header row (assuming it contains column names)
#     headers = next(csv_reader)
    
#     for row in csv_reader:
#         # Insert data into the table
#         query = f"INSERT INTO heart ({', '.join(headers)}) VALUES ({', '.join(['?']*len(headers))})"
#         cursor.execute(query, row)

# # Commit changes and close connection
# conn.commit()
# conn.close()
import sqlite3
import pandas as pdw
conn = sqlite3.connect('C:\\AIhealthpro\\instance\\AIhealthpro.db')  # Replace 'your_database.db' with your database file

# Query your data from the database into a DataFrame
query = "SELECT Age ,Gender,Family_history,Blood_Pressure, Hypertension,Smoking ,Stress,Alcoholic,BodyWeight,Excessive_intakeof_salt,Excessive_intakeof_coffee,result FROM heart"  # Replace 'your_table' with your table name
data = pd.read_sql_query(query, conn)
print(data)
