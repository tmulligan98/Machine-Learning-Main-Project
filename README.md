# Machine Learning Main Project


## Pre-Processing
### For a given excel "Multi-Day Volume by Direction..."
1. Remove rows 1-6
2. Remove Columns E-M
3. Remove Events at bottom
4. Ensure Excel is only from start to end of month (no overlap of days)
5. Save as CSV to a folder for that year
6. Clean up extra rows, these will break the pre-processing script. (Look at the saved csv)
7. When saving the resultant csv, specify the correct file name and location

### To-Do
Need to validate data to ensure that each column contains what we expect.
Some date entries are funky. I think these are days where the sensors aren't working. Empty entries appear as '-' from the data provider. We should probably convert these to NaNs