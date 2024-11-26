# fraud-detection
Fraud Detection Pipeline Using Kafka and Machine Learning
This project implements a real-time fraud detection pipeline using Apache Kafka, machine learning, and MySQL. The goal of this project is to monitor transactions and predict whether they are fraudulent using a machine learning model. The pipeline is built to process streaming data from Kafka, apply preprocessing and transformations, and then use a trained Random Forest model to classify transactions as fraudulent or not. The results are stored in a MySQL database for further analysis.

Table of Contents
Overview
Features
Technologies Used
Project Setup
How It Works
Model Training and Prediction
Contributing
License
Overview
In this project, I’ve built an end-to-end machine learning pipeline for detecting fraud in real-time using Kafka for stream processing and MySQL for storing results. The pipeline receives incoming transaction data, processes it using feature engineering techniques, predicts fraud using a trained machine learning model, and stores the results in a database.

The core components of the pipeline include:

Kafka Producer: Simulates a stream of transaction data.
Kafka Consumer: Reads the incoming transaction data and processes it.
Data Preprocessing: Includes feature engineering and scaling of data.
Machine Learning Model: A Random Forest classifier is trained to predict fraudulent transactions.
MySQL Database: Stores the results of the fraud detection predictions.
Features
Real-time Fraud Detection: Utilizes Kafka for real-time message streaming and a Random Forest classifier for fraud prediction.
Automated Data Ingestion: Kafka consumes streaming transaction data and applies the model automatically.
Data Storage: The predicted fraud results are stored in MySQL for future analysis.
Scalable Architecture: Kafka allows for handling large volumes of transaction data in a distributed system.
Model Training: The Random Forest model is trained on historical data and predicts the likelihood of fraud in transactions.
Technologies Used
Apache Kafka: For real-time data streaming and message consumption.
Python: For implementing the data processing pipeline and machine learning model.
Scikit-learn: For building and training the machine learning model.
imblearn (SMOTE): For handling class imbalance during training.
MySQL: For storing transaction data and fraud predictions.
pandas: For data manipulation and preprocessing.
geopy: For calculating distances between locations as a feature in the model.
Random Forest Classifier: Used as the predictive model for fraud detection.
Project Setup
Prerequisites
Install Apache Kafka and set up a Kafka cluster.
Install MySQL for storing the results of fraud predictions.
Install Python 3.x and the required libraries.
Install Dependencies
Clone the repository to your local machine:

bash
Copy code
git clone https://github.com/your-username/fraud-detection-pipeline.git
Navigate to the project directory and install required dependencies:

bash
Copy code
pip install -r requirements.txt
Ensure Kafka and MySQL are running locally or on a server and configure the connection details as needed.

Kafka Setup
Create a Kafka topic (e.g., my-topic-2) to receive transaction data:

bash
Copy code
kafka-topics.sh --create --topic my-topic-2 --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1
Start Kafka Producer to simulate transactions:

Producer sends transaction data to the Kafka topic.
Example data will include fields like lat, long, category, amt, transaction_id, etc.
MySQL Setup
Create a MySQL database:

sql
Copy code
CREATE DATABASE kafka_db;
Create a table to store transactions:

sql
Copy code
CREATE TABLE transactions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    transaction_id VARCHAR(255) NOT NULL,
    fraud INT DEFAULT 0
);
Modify the database connection parameters (username, password, port) in the Python code to match your local MySQL setup.

How It Works
Producer: A Kafka producer sends simulated transaction data (including features like lat, long, amt, category, transaction_id, etc.) to the Kafka topic.
Consumer: A Kafka consumer reads messages from the topic and passes the transaction data to the fraud detection pipeline.
Preprocessing: The pipeline applies data preprocessing, including feature engineering (e.g., calculating the distance between lat/long and merchant lat/long) and scaling the numerical features.
Prediction: The trained machine learning model (Random Forest Classifier) predicts whether the transaction is fraudulent (1) or not (0).
Storing Results: The prediction is stored in the MySQL database, along with the transaction ID and fraud status.
Model Training and Prediction
Training the Model: The model is trained using historical transaction data. This data includes labeled information indicating whether a transaction is fraudulent or not. We use Random Forest Classifier to train the model.

Prediction: After training, the model is used to predict fraud on incoming transactions, and the results are output to a MySQL database.

SMOTE (Synthetic Minority Over-sampling Technique): To address class imbalance (where fraud cases are much less frequent than non-fraud cases), we use SMOTE to generate synthetic fraud cases during the model training phase.

Contributing
Contributions are always welcome! If you'd like to improve this project, please feel free to open an issue or submit a pull request.
