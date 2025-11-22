# How to Run the KNN Classification Script

In this project, I worked with a simple CSV file and applied the **KNN**
algorithm to see whether the output would be classified as iPhone or
Android.
# Requirements
You need to have Python installed (any recent version works).\
Then install the required libraries:

``` bash
pip install pandas numpy scikit-learn
```
# Project Structure
The project files are organized like this:

    project_folder
     DATA CSV.csv
     script.py
     README.md
- **DATA CSV.csv**: The dataset used for the classification\
- **script.py**: The code that trains the model and makes predictions\
- **README.md**: This explanation written in a simple style
# How to Run the Script
Open your terminal, navigate to the project folder, and run:

``` bash
python script.py
```
# What the Script Does
1.  Reads the data from the CSV file.
2.  Cleans and converts the values into numerical format so the model
    can process them.
3.  Splits the data into training and testing sets.
4.  Trains a KNN model with 5 neighbors.
5.  Calculates and displays the model's accuracy.
6.  Takes a sample from the test data and predicts whether the
    preference is iPhone or Android.
