# Import modules
from docquery import document, pipeline
import sys
import torch
#---------------


# Fetch file path for file to load.
file_path = sys.argv[1]

# Initialize the document reading pipeline.
doc_pipeline = pipeline("document-question-answering")

# Load the file.
document = document.load_document(file_path)

# Select which processor to run the model on.
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Ask questions.
while True:
    question = input("Enter your question: ")
    print(question, doc_pipeline(question = question, **document.context))
