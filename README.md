**MLOps in Action: A Reproducible Salary Prediction Pipeline**

This project demonstrates a complete, end-to-end MLOps pipeline to predict individual salaries in India using the "Gender Pay Gap" dataset. The core focus is on creating a reproducible, version-controlled, and automated machine learning system using modern MLOps tools.

**🚀 MLOps Workflow**
This project is built around a robust, automated workflow that ensures reproducibility and continuous integration.

Version Control: We use a dual version control system. Git tracks our Python source code, while DVC tracks our large data and model files, keeping the repository lightweight.

Reproducible Pipeline: The entire workflow is defined in dvc.yaml. This file orchestrates all stages, from data preprocessing to model evaluation, ensuring that anyone can reproduce the results with a single command.

Experiment Tracking: MLflow is integrated into our pipeline to automatically log the parameters and performance metrics of every model we train. This provides an interactive dashboard to compare experiments.

CI/CD Automation: A GitHub Actions workflow is configured to automatically run the entire DVC pipeline on every git push. This ensures our model is continuously validated against the latest code and data.

**🛠️ Tech Stack**
Language: Python 3.9

Data Analysis: Pandas, NumPy

Machine Learning: Scikit-learn

Version Control: Git & DVC (Data Version Control)

Experiment Tracking: MLflow

Automation (CI/CD): GitHub Actions

Remote Storage: AWS S3

**📂 Project Structure**
├── .github/workflows/      # CI/CD pipeline definition
├── data/
│   ├── raw/                # Raw, immutable data
│   └── processed/          # Cleaned data (Tracked by DVC)
├── models/                 # Trained models (Tracked by DVC)
├── notebooks/              # Jupyter notebooks for EDA
├── src/                    # Source code for the pipeline stages
├── .dvc/                   # DVC internal files
├── .gitignore              # Files to be ignored by Git
├── dvc.yaml                # DVC pipeline definition
├── metrics.json            # Final model metrics (Tracked by DVC)
├── params.yaml             # Parameters for the pipeline
├── README.md               # You are here!
└── requirements.txt        # Project dependencies
⚙️ Setup and Installation
To set up this project on your local machine, follow these steps.

**Clone the Repository**

Bash

git clone https://github.com/KunalChitroda/salary_pay_gap.git
cd salary_pay_gap
Create and Activate a Virtual Environment

Bash

python -m venv venv
venv\Scripts\activate  # On Windows
# source venv/bin/activate  # On macOS/Linux
Install Dependencies

Bash

pip install -r requirements.txt
Configure DVC with Your Remote Storage
If you're using AWS S3, configure your credentials:

Bash

aws configure
Pull DVC-tracked Data
This command downloads the dataset and models from the remote S3 storage.

Bash

dvc pull
▶️ How to Run the Pipeline
You can reproduce the entire pipeline or run new experiments with DVC.

Reproduce the Pipeline
This command will run all the stages defined in dvc.yaml (preprocess, train, evaluate).

Bash

dvc repro
Run a New Experiment
To train the model with different parameters, use dvc exp run with the --set-param flag. DVC will run the experiment without overwriting your main results.

Bash

dvc exp run --set-param train.n_estimators=150

View Experiments
To see a table comparing all your experiments, run:

Bash

dvc exp show --only-changed

**📊 Experiment Tracking with MLflow**
All experiments are logged with MLflow. To launch the interactive dashboard, run:

Bash

mlflow ui
Then open your browser to http://127.0.0.1:5000 to view and compare your runs.

📈 Results
Our Random Forest model was trained on a fixed set of hyperparameters defined in params.yaml. The final performance on the test set is:

R-squared Score: ~96.98%

This indicates a strong predictive performance. The metrics for the current version of the model are always available in the metrics.json file.
