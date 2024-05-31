import pandas as pd

def load_data(uploaded_file):
    try:
        file_name = uploaded_file.name
        #Check if the file is a CSV
        if file_name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        #Check if the file is an Excel file
        elif file_name.endswith('.xlsx'):
            data = pd.read_excel(uploaded_file)
        #Raise an error for unsupported file formats
        else:
            raise ValueError("Unsupported file format")
        return data
    except Exception as e:
        #Return the error message as a string
        return str(e)