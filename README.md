# Blooms Taxonomy Classification

This project classifies text data according to Bloom's Taxonomy levels using machine learning techniques. It includes a Flask server for API access, PDF text extraction, and various scripts for data processing and model training.

## Project Structure

- `main.py` - Core logic for training, predicting, and processing data.
- `server.py` - Flask API server for model interaction.
- `pdftotext.py` - Extracts text from PDF files.
- `int.py` - Installs required Python packages.
- `model.joblib` - Serialized trained model.
- `dataset/` - Contains CSV and Excel files with taxonomy data.
- `materials/` - Additional resources.
- `other/` - Experimental and utility scripts.
- `test/` - Test input files.

## Requirements

Install dependencies using:

```sh
pip install -r requirements.txt
```

## Usage

### 1. Train the Model

Run the following to train and save the model:

```sh
python main.py
```

### 2. Start the Server

Start the Flask API server:

```sh
python server.py
```

The server will be available at `http://localhost:3100/`.

### 3. API Endpoints

- `POST /all` - Full processing pipeline.
- `POST /train-model` - Retrain the model.
- `POST /predict` - Predict taxonomy levels from a PDF file.

## PDF Text Extraction

The project uses `pdfminer` and NLTK to extract and preprocess text from PDF files.

## Data

Datasets are located in the `dataset/` directory. Update or add your own data as needed.

## License

This project is for educational and research purposes.
