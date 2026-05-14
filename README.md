Contains 2 different datasets and a code to run the linear regression.\n
the 2020_housePrices.csv dataset contains resale HDB details from 2020 only.
the Resale flat prices based on registration date from Jan-2017 onward.csv dataset contains resale HDB details from 2017 onwards.
In the linear_regression2.py, u can choose between each files to run the model by commenting out the other datafile.

Results: 2 different dataset produces 2 different levels of loss, might have to due with not taking into account of the year the resale price is sold as there is 
non-linear relationship between inflation and year which may result in higher loss when the model is learning from multiple years.
