import pandas as pd

def load_data(uploaded_file):
    try:
        file_name = uploaded_file.name
        if file_name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        elif file_name.endswith('.xlsx'):
            data = pd.read_excel(uploaded_file)
        else:
            raise ValueError("Unsupported file format")
        return data
    except Exception as e:
        return str(e)