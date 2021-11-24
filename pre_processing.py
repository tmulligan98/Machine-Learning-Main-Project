import pandas as pd
import os
from datetime import datetime

#Data taken from M50 Between Jn06 N03/M50 and Jn05 N02/M50, Finglas, Co. Dublin:
#https://trafficdata.tii.ie/sitedashboard.asp?sgid=XZOA8M4LR27P0HAO3_SRSB&spid=281080600524

def create_path(dir, folder):
    path = f"{dir}/{folder}"
    if os.path.exists(dir):
        if not os.path.exists(path):
            print("Created: ",path)
            os.mkdir(path)
        else:
            print(f"{path} already exists")
    else: 
        print(f"{dir} doesnt exists")

months_of_year = ['Jan', 'Feb', 'Mar', 'Apr', 
            'May', 'Jun', 'Jul', 'Aug', 
            'Sep', 'Oct', 'Nov', 'Dec']

years = [2014, 2015, 2016, 2017, 2018]
site_location = "M50 Between Jn06 N03/M50 and Jn05 N02/M50, Finglas, Co. Dublin"

for current_year in years:

    dir = "Machine-Learning-Main-Project/Data/Data"+str(current_year)
    create_path(dir, "formatted")
    create_path(dir, "preprocessed")
    
    for current_month in months_of_year:
        xl = pd.ExcelFile(dir+f"/raw/{current_month}.xlsx")
        cols=[0,1,2,3,4]
        df = xl.parse("Multi-Day Volume by Direction", skiprows=6, usecols=cols)
        print(df.head())

        num_rows = 0
        for row in df.iloc:
            if (row.Date == 'Events'):
                event_row = num_rows-1
            num_rows = num_rows+1

        df = df.drop(labels=range(event_row, num_rows), axis=0)

        output_file_path = os.path.join(dir+f'/formatted', f"{current_month}.csv") 
        df.to_csv(output_file_path, index=False)

        #We have saved this formatted data in a new file
        #Now lets process the data to suit out application

        days = {"monday":1, "tuesday":2, "wednesday":3, "thursday":4, 
                "friday":5, "saturday":6, "sunday":7}

        months = {"january" : 1, "february": 2, "march" : 3, "april" : 
                    4, "may" : 5, "june" : 6, "july" : 7, "august" : 8, 
                    "september" : 9, "october" : 10, "november" : 11, "december" : 12}
        # SPECIFY THE LOCATION OF THE DATA HERE
        location = site_location

        # Preprocess the data and extract some basic features
        month = 0
        day = 0
        datetime_string = ""
        date_string = ""
        day_of_month_string = ""
        month_string = ""
        year_string = ""
        data_point_south  = []
        data_point_north = []
        output_df = pd.DataFrame()
        temp_df : pd.DataFrame()
        column_names = ["datetime", "dayOfWeek", "month", "time", "northBound", "southBound"]


        for index, row in df.iterrows():

            # Check if 'Total' exists in time column of current row,
            # if it does skip this iteration.
            if row["Time"] == "Total":
                continue
            # Check if the current row has a day specified
            if not pd.isna(row["Date"]):

                # get the full date
                date_string = row["Date"]

                # Get the day, month and year as a string
                day_of_month_string = date_string.split()[1].lower()
                month_string = months[date_string.split()[2].lower()]
                year_string = date_string.split()[3].lower()

                # Some basic features
                day = int(days[date_string.split()[0].lower()])
                month = int(month_string)

                # Put together the date as a string
                date_string = f"{year_string}/{month_string}/{day_of_month_string}"

            # Get the details to add to our new csv
            time = str(row["Time"])
            datetime_string = f"{date_string} {time}"
            time = time.replace(":","")
            data_point_north = [datetime_string, day, month, int(time), row["N"], row["S"]]
            # data_point_south = [date_string, time, 1, row["S"]] We might have to do separate data sets. Time series predictions don't like other data
            temp_df = pd.DataFrame([data_point_north], columns=column_names)
            output_df = output_df.append(temp_df, ignore_index=True)   

        print("Preprocessed data...")
        print(output_df.head())
        output_file_path = os.path.join(dir+f'/preprocessed', f'{current_month}.csv')
        output_df.to_csv(output_file_path, index=False)