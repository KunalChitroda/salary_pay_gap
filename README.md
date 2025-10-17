# MLOps in Action: A Customer Churn Prediction API

This project demonstrates a complete, end-to-end MLOps pipeline to predict customer churn. The core focus is on creating a reproducible, version-controlled, and automated machine learning system that is ultimately deployed as a live prediction API.

-----

## 🚀 MLOps Workflow

This project is built around a robust, automated workflow that ensures reproducibility, continuous integration, and continuous delivery.

1.  **Version Control**: We use a dual version control system. **Git** tracks our Python source code, while **DVC** tracks our data and model files, keeping the repository lightweight.
2.  **Reproducible Pipeline**: The entire training workflow is defined in `dvc.yaml`. This file orchestrates all stages, from data preprocessing to model evaluation, ensuring that anyone can reproduce the model with a single command (`dvc repro`).
3.  **Experiment Tracking**: **MLflow** is integrated into our pipeline to automatically log the parameters and performance metrics of every model we train. This provides an interactive dashboard to compare experiments.
4.  **CI/CD Automation**: A **GitHub Actions** workflow is configured to automatically run the entire DVC training pipeline on every `git push`. This ensures our model is continuously validated against the latest code.
5.  **Deployment**: The trained model is served via a **FastAPI** application, which is containerized using **Docker** for easy and consistent deployment.

-----

## 🛠️ Tech Stack

  * **Language**: Python 3.9
  * **Machine Learning**: Scikit-learn, Pandas
  * **Version Control**: Git & DVC (Data Version Control)
  * **Experiment Tracking**: MLflow
  * **API Framework**: FastAPI
  * **Containerization**: Docker
  * **Automation (CI/CD)**: GitHub Actions
  * **Remote Storage**: AWS S3

-----

## 📂 Project Structure

```
├── .github/workflows/      # CI/CD pipeline definition
├── data/
│   ├── raw/                # Raw, immutable data
│   └── processed/          # Cleaned data (Tracked by DVC)
├── models/                 # Trained models (Tracked by DVC)
├── src/                    # Source code for the DVC pipeline stages
├── .dvc/                   # DVC internal files
├── .gitignore              # Files to be ignored by Git
├── app.py                  # FastAPI application for serving predictions
├── Dockerfile              # Docker instructions for building the API image
├── dvc.yaml                # DVC pipeline definition
├── metrics.json            # Final model metrics (Tracked by DVC)
├── params.yaml             # Parameters for the pipeline
├── README.md               # You are here!
└── requirements.txt        # Project dependencies
```

-----

## ⚙️ Setup and Installation

To set up this project on your local machine, follow these steps.

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/KunalChitroda/salary_pay_gap.git
    cd salary_pay_gap
    ```
2.  **Create and Activate a Virtual Environment**
    ```bash
    python -m venv venv
    venv\Scripts\activate  # On Windows
    # source venv/bin/activate  # On macOS/Linux
    ```
3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Configure DVC with Your Remote Storage**
    If you're using AWS S3, configure your credentials:
    ```bash
    aws configure
    ```
5.  **Pull DVC-tracked Data**
    This command downloads the dataset and trained model from the remote S3 storage.
    ```bash
    dvc pull
    ```

-----

## ▶️ How to Run the Project

You can either reproduce the training pipeline or run the live prediction API.

### Reproduce the Training Pipeline

This command will run all the stages defined in `dvc.yaml` (preprocess, train, evaluate).

```bash
dvc repro
```

### Run the Prediction API Locally

This command starts the FastAPI server.

```bash
uvicorn app:app --reload
```

Then open your browser to **`http://127.0.0.1:8000/docs`** to interact with the API.

-----

## 🐳 Docker Deployment

The prediction API is designed to be deployed as a Docker container.

1.  **Build the Docker Image**
    ```bash
    docker build -t churn-api .
    ```
2.  **Run the Docker Container**
    ```bash
    docker run -p 8080:80 churn-api
    ```
    The API will then be accessible at **`http://localhost:8080/docs`**.

-----

## 📊 Experiment Tracking with MLflow

All training runs are logged with MLflow. To launch the interactive dashboard, run:

```bash
mlflow ui
```

Then open your browser to `http://127.0.0.1:5000` to view and compare your runs.

-----

## 📈 Results

Our Random Forest Classifier was trained on a fixed set of hyperparameters defined in `params.yaml`. The final performance on the test set is:

  * **Accuracy**: \~79.60%
  * **Precision**: \~66.29%
  * **Recall**: \~47.33%

This indicates a solid baseline performance for identifying churning customers. The metrics for the current version of the model are always available in the `metrics.json` file.