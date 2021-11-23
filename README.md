# Machine Learning Main Project


## Pre-Processing
1. Download raw data for TII for each months 'Multi day volume by direction' for a given year
2. Label each .xlsx file for the given months (Jan-Dec)
3. Add these files to the directory Data/Data{year}/raw, where year is the year of the downloaded data
4. Add the year to the years list variable in pre_processing.py
5. Run pre_processing.py, this will format and perform the pre processing on the raw data, this new data will be placed inside .csv files in the years directory.

### To-Do
Need to validate data to ensure that each column contains what we expect.
Some date entries are funky. I think these are days where the sensors aren't working. Empty entries appear as '-' from the data provider. We should probably convert these to NaNs