# Disaster Response Pipeline Project
## Table of Contents
1. [Project Overview](#po)
2. [Project Components](#pc)
3. [Project Objective](#pob)
4. [Build With](#bw)
5. [Instructions](#in)
6. [Credits and Acknowledgements](#ca)

<a name="po"></a>
## 1. Project Overview
This project is part of Udacity Nano Degree course assignment focusing on data engineering skills. Disaster data from Figure Eight is used to analyze  and build a model for an API that classifies disaster messages.

There is a data set containing real messages that were sent during disaster events and machine learning pipeline is created to categorize these events so that the messages can be sent to an appropriate disaster relief agency.

Your project will include a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data. 
<a name="pc"></a>
## 2. Project Components
There are three components in this project.
### 2.1 ETL Pipeline
In Python script, process_data.py is served as loader of the messages and categories datasets. It then merge the two datasets and cleans it. Finally, it will stores it in a SQLite database. Similar detailed step by step approach are availabel in Notebook folder under ETL file name. 
### 2.2 ML Pipeline
In a Python script, train_classifier.py is served as machine learning pipeline that loads data from the SQLite database, splits the dataset into training and test sets, builds a text processing and machine learning pipeline, trains and tunes a model using GridSearchCV, outputs results on the test set, and exports the final model as a pickle file.Similar detailed step by step approach are availabel in Notebook folder under ML file name. 
### 2.3 Flask Web App
The final output will be displayed in a flask web app with addition of two data visualizations made with Plotly. 
<a name="pob"></a>
## 3. Project Objective
This code is designed to iniate a web app which an emergency operators could exploit during a disaster (e.g. an earthquake or Tsunami), to classify a disaster text messages into several categories which then can be transmited to the responsible entity
<a name="bw"></a>
## 4. Built With
* Python 3+
* Machine Learning Libraries: NumPy, SciPy, Pandas, Sciki-Learn
* Natural Language Process Libraries: NLTK
* SQLlite Database Libraqries: SQLalchemy
* Web App and Data Visualization: Flask, Plotly
<a name="in"></a>
## 5. Instructions
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
<a name="ca"></a>
## 6. Credits and Acknowledgements
* [Udacity](https://www.udacity.com/) for providing such a complete Data Science Nanodegree Program
* [Figure Eight](https://www.figure-eight.com/) for providing messages dataset to train my model

