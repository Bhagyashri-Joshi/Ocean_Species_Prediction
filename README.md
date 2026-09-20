# Ocean Penguin Species Predictor

## Project Overview
An interactive Streamlit machine learning application that predicts penguin species from physical measurements and observation details.

## Features
- Loads the Palmer Penguins dataset.
- Preprocesses numerical and categorical features.
- Trains a Random Forest classification model.
- Displays test accuracy.
- Accepts user inputs through Streamlit widgets.
- Displays predicted species and class probabilities.
- Shows dataset preview and classification metrics.

## Tech Stack
- Python
- Streamlit
- Pandas
- Scikit-learn
- Random Forest Classifier

## Project Structure
```text
ocean_penguin_streamlit_app/
├── app.py
├── requirements.txt
└── README.md
```

## Installation and Setup
1. Install Python 3.9 or above.
2. Open a terminal in the project directory.
3. Create a virtual environment:

```bash
python -m venv venv
```

4. Activate the environment on Windows:

```bash
venv\Scripts\activate
```

5. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage
Run the application using:

```bash
streamlit run app.py
```

The application will open in your browser. Enter the penguin details and click **Predict Species**.

## Model
The application uses a Random Forest Classifier. Numerical features are imputed using the median, while categorical features are imputed using the most frequent value and encoded using OneHotEncoder.

## Limitations
- The application depends on internet access to download the dataset.
- The dataset contains only three penguin species.
- Predictions are educational and should not be used for scientific or conservation decisions.
- The model's performance depends on the quality and distribution of the dataset.

## Future Improvements
- Add visualizations such as feature distributions.
- Add model comparison with Logistic Regression and Decision Trees.
- Store the trained model in a separate file.
- Add a conservation-focused dataset and prediction task.
- Deploy the application using Streamlit Community Cloud.
