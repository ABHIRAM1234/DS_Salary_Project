# Project Overview - Data Science Salary Predictor
#### This is my first Major Data Science Project
- Created a tool that estimates the salary of a data scientist to help data scientist negotiate their income when they get a job
- Scrapped more than 1000 job descriptions from (https://www.glassdoor.co.in/index.htm "Glassdoor")
- Performed data cleaning and feature engineerin from the text of each job description to quantify the value companies put on different data science tools like python, excel, aws, and spark.
- Created different models by optimizing Linear, Lasso, and Random Forest Regressors using GridsearchCV to reach the best model.
- Built an API using Flask

## Code and Resources Used

**Python Version** : 3.8
**Packages:** numpy, pandas, sklearn, matplotlib, seaborn, selenium, flask, json, pickle  
**For Web Framework Requirements:**  ```pip install -r requirements.txt```  
**Scraper Github:** https://github.com/arapfaik/scraping-glassdoor-selenium  
**Scraper Article:** https://towardsdatascience.com/selenium-tutorial-scraping-glassdoor-com-in-10-minutes-3d0915c6d905  
**Flask Productionization:** https://towardsdatascience.com/productionize-a-machine-learning-model-with-flask-and-heroku-8201260503d2

## Web Scraping
Made some changes to the web scraper github repo (above) to scrape more than 1000 job postings from glassdoor. For each job, we got the following:

*	Job title
*	Salary Estimate
*	Job Description
*	Rating
*	Company 
*	Location
*	Company Headquarters 
*	Company Size
*	Company Founded Date
*	Type of Ownership 
*	Industry
*	Sector
*	Revenue
*	Competitors 

## Data Cleaning
After scraping the data, I needed to clean it up so that it will be able to be fed into the model. I made the following changes and created the following variables:

*	Parsed numeric data out of salary 
*	Made columns for employer provided salary and hourly wages 
*	Removed rows without salary 
*	Parsed rating out of company text 
*	Made a new column for company state 
*	Added a column for if the job was at the company’s headquarters 
*	Transformed founded date into age of company 
*	Made columns for if different skills were listed in the job description:
    * Python  
    * R  
    * Excel  
    * AWS  
    * Spark 
*	Column for simplified job title and Seniority 
*	Column for description length 

## Exploratory Data Analysis
I looked at the distributions of the data and the value counts for the various categorical variables. Below are a few highlights from the pivot tables. 

![alt text](https://github.com/ABHIRAM1234/DS_Salary_Project/blob/main/Job%20Opportunities%20by%20State.PNG "Job Opportunites in different states")
![alt text](https://github.com/ABHIRAM1234/DS_Salary_Project/blob/main/Salary_by_position.PNG "Salary for Different Position")
![alt text](https://github.com/ABHIRAM1234/DS_Salary_Project/blob/main/correlation.PNG "Correlations")

## Model Building 

First, I transformed the categorical variables into dummy variables. I also split the data into train and tests sets with a test size of 20%.   

I tried three different models and evaluated them using Mean Absolute Error. I chose MAE because it is relatively easy to interpret and outliers aren’t particularly bad in for this type of model.   

I tried three different models:
*	**Multiple Linear Regression** – Baseline for the model
*	**Lasso Regression** – Due to the sparse data from many categorical variables, I thought a normalized regression like lasso would be effective.
*	**Random Forest** – Again, with the sparsity associated with the data, I thought that this would be a good fit. 

## Model performance
The Random Forest model far outperformed the other approaches on the test and validation sets. 
*	**Random Forest** : MAE = 7.22
*	**Multiple Linear Regression**: MAE = 18.86
*	**Lasso Regression**: MAE = 19.67
*	**Test Ensemble**: Ensemble methods are techniques that create multiple models and then combine them to produce improved results

## Productionization 
I built a flask API endpoint that was hosted on a local web server by following along with the tutorial in the reference section above. The API endpoint takes in a request with a list of values from a job listing and returns an estimated salary.
