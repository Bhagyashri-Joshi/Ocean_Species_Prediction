import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

st.set_page_config(
    page_title="Ocean Penguin Species Predictor",
    page_icon="🐧",
    layout="centered"
)

st.title("Ocean Penguin Species Predictor")
st.write(
    "This application predicts a penguin species using physical measurements "
    "and basic observation details."
)


@st.cache_data
def load_data():
    """Load the Palmer Penguins dataset."""
    url = (
        "https://raw.githubusercontent.com/allisonhorst/"
        "palmerpenguins/main/inst/extdata/penguins.csv"
    )
    return pd.read_csv(url)


@st.cache_resource
def train_model(data):
    """Preprocess data, train the model, and calculate evaluation metrics."""

    features = [
        "bill_length_mm",
        "bill_depth_mm",
        "flipper_length_mm",
        "body_mass_g",
        "island",
        "sex"
    ]

    target = "species"

    # Keep only the required columns and remove incomplete rows.
    model_data = data[features + [target]].dropna()

    X = model_data[features]
    y = model_data[target]

    numerical_features = [
        "bill_length_mm",
        "bill_depth_mm",
        "flipper_length_mm",
        "body_mass_g"
    ]

    categorical_features = ["island", "sex"]

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numerical", numerical_pipeline, numerical_features),
        ("categorical", categorical_pipeline, categorical_features)
    ])

    classifier = RandomForestClassifier(
        n_estimators=150,
        random_state=42
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    # Split the dataset into training and testing sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Train the model only on the training set.
    pipeline.fit(X_train, y_train)

    # Calculate training and testing predictions.
    train_predictions = pipeline.predict(X_train)
    test_predictions = pipeline.predict(X_test)

    train_accuracy = accuracy_score(y_train, train_predictions)
    test_accuracy = accuracy_score(y_test, test_predictions)

    # Perform 5-fold cross-validation on the complete cleaned dataset.
    cv_scores = cross_val_score(
        pipeline,
        X,
        y,
        cv=5,
        scoring="accuracy"
    )

    # Classification report and confusion matrix for the held-out test set.
    report = classification_report(
        y_test,
        test_predictions,
        output_dict=True,
        zero_division=0
    )

    labels = sorted(y.unique())
    matrix = confusion_matrix(
        y_test,
        test_predictions,
        labels=labels
    )

    confusion_df = pd.DataFrame(
        matrix,
        index=[f"Actual: {label}" for label in labels],
        columns=[f"Predicted: {label}" for label in labels]
    )

    return {
        "pipeline": pipeline,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "cv_scores": cv_scores,
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "report": report,
        "confusion_matrix": confusion_df,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "features": features,
        "classes": list(pipeline.named_steps["classifier"].classes_)
    }


try:
    data = load_data()
    results = train_model(data)

    model = results["pipeline"]

    # Sidebar: model information and evaluation.
    st.sidebar.header("Model Information")
    st.sidebar.write("Algorithm: Random Forest Classifier")
    st.sidebar.write(f"Training samples: {results['train_size']}")
    st.sidebar.write(f"Testing samples: {results['test_size']}")

    st.sidebar.subheader("Evaluation")
    st.sidebar.metric(
        "Training Accuracy",
        f"{results['train_accuracy'] * 100:.2f}%"
    )
    st.sidebar.metric(
        "Test Accuracy",
        f"{results['test_accuracy'] * 100:.2f}%"
    )
    st.sidebar.metric(
        "Mean 5-Fold CV Accuracy",
        f"{results['cv_mean'] * 100:.2f}%"
    )

    st.sidebar.write(
        f"CV Standard Deviation: {results['cv_std'] * 100:.2f}%"
    )

    st.subheader("Enter Penguin Details")

    bill_length = st.number_input(
        "Bill Length (mm)",
        min_value=25.0,
        max_value=65.0,
        value=45.0,
        step=0.1
    )

    bill_depth = st.number_input(
        "Bill Depth (mm)",
        min_value=10.0,
        max_value=25.0,
        value=17.0,
        step=0.1
    )

    flipper_length = st.number_input(
        "Flipper Length (mm)",
        min_value=150,
        max_value=250,
        value=200,
        step=1
    )

    body_mass = st.number_input(
        "Body Mass (g)",
        min_value=2500,
        max_value=7000,
        value=4000,
        step=50
    )

    island = st.selectbox(
        "Island",
        options=["Biscoe", "Dream", "Torgersen"]
    )

    sex = st.selectbox(
        "Sex",
        options=["Male", "Female"]
    )

    if st.button("Predict Species"):
        input_data = pd.DataFrame([{
            "bill_length_mm": bill_length,
            "bill_depth_mm": bill_depth,
            "flipper_length_mm": flipper_length,
            "body_mass_g": body_mass,
            "island": island,
            "sex": sex
        }])

        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        classes = results["classes"]

        st.success(f"Predicted Penguin Species: {prediction}")

        probability_df = pd.DataFrame({
            "Species": classes,
            "Probability (%)": (probabilities * 100).round(2)
        })

        st.subheader("Prediction Probabilities")
        st.dataframe(
            probability_df,
            width='stretch',
            hide_index=True
        )

    with st.expander("View Dataset Preview"):
        st.dataframe(
            data.head(10),
            width='stretch',
            hide_index=True
        )

    with st.expander("View Cross-Validation Results"):
        cv_df = pd.DataFrame({
            "Fold": range(1, len(results["cv_scores"]) + 1),
            "Accuracy (%)": (results["cv_scores"] * 100).round(2)
        })

        st.dataframe(
            cv_df,
            width='stretch',
            hide_index=True
        )

        st.write(
            f"Mean CV Accuracy: {results['cv_mean'] * 100:.2f}%"
        )
        st.write(
            f"CV Standard Deviation: {results['cv_std'] * 100:.2f}%"
        )

    with st.expander("View Classification Report"):
        metrics_df = pd.DataFrame(results["report"]).transpose()
        st.dataframe(
            metrics_df,
            width='stretch'
        )

    with st.expander("View Confusion Matrix"):
        st.dataframe(
            results["confusion_matrix"],
            width='stretch'
        )

except Exception as error:
    st.error(
        "The application could not load the dataset or train the model. "
        "Check your internet connection and installed dependencies."
    )
    st.exception(error)