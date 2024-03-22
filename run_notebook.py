from nbconvert import PythonExporter

# Define the path to the notebook file
notebook_path = "/Users/shubhamgaur/Desktop/HackPOC.ipynb"

# Create a PythonExporter instance
exporter = PythonExporter()

# Read the notebook and export it as a Python script
with open(notebook_path) as f:
    python_script, _ = exporter.from_file(f)

# Execute the Python script
exec(python_script)
