"""
Created on Sat Dec 03 01:15:12 2022

@author: Srinivas
"""

import pandas as pd # pandas library used for Data Manipulation and Analysis
import numpy as np # numpy library for mathematical operations
import matplotlib.pyplot as plt # Library to visualise data
import seaborn as sns # Library to visualise data (making statistical graphics)
import requests # For sending HTTP requests to web
from IPython.display import display # For displaying functions
from tabulate import tabulate  # Library to visualise data in tabular format

# Base URL used in all the API calls
base_url = 'http://api.worldbank.org/v2/'

# Indicators list according to the features 
indic_codes = ['SP.POP.TOTL',
                   'SP.POP.TOTL.FE.IN',
                   'SP.POP.TOTL.MA.IN',
                   'SL.AGR.EMPL.ZS', 
                   'SL.AGR.EMPL.FE.ZS',
                   'SL.IND.EMPL.FE.ZS',
                   'NY.GDP.MKTP.CD',
                   'NV.AGR.TOTL.CD',
                   'EG.USE.ELEC.KH.PC', 
                   'EG.FEC.RNEW.ZS', 
                   'EG.USE.COMM.FO.ZS']


countries = ['USA', 'India', 'China', 'Japan', 'Canada', 'Great Britain', 'South Africa']

# mapping of codes of feature to more meaningful names
feature_map = {
    "SP.POP.TOTL": "Total Population",
    "SP.POP.TOTL.FE.IN": "Female Population",
    "SP.POP.TOTL.MA.IN": "Male Population", 
    "SL.AGR.EMPL.ZS": "Employment in Agriculture(%)",
    "SL.AGR.EMPL.FE.ZS": "Female Employment in Agriculture(%)",
    "SL.IND.EMPL.FE.ZS": "Female Employment in Industry(%)",
    "NY.GDP.MKTP.CD": "GDP in USD",
    "NV.AGR.TOTL.CD":"Agriculture value added(in USD)",
    "EG.USE.ELEC.KH.PC":"Electric Power Consumption(kWH per capita)",
    "EG.FEC.RNEW.ZS":"Renewable Energy Consumption (%)",
    "EG.USE.COMM.FO.ZS":"Fossil Fuel Consumption (%)"
}

# Mapping of country codes to their real/full names
country_map = {
    "US": "USA",
    "IN":"India",
    "CN": "China",
    "JP": "Japan",
    "CA": "Canada",
    "GB": "Great Britain",
    "ZA": "South Africa"
}

# sending the request using constant parameters.
params = dict()
# ensuring to receive a JSON response
params['format'] = 'json'
# The data I'm fetching is for 59 years.
# Hence changing the default page size of 50 to 100 to call one API per feature.
params['per_page'] = '100'
# To get data defining Range of years
params['date'] = '1960:2018'


"""
Fetching data through API calls:
    
  The following functions have been written to make an API call per feature and fetch the data.
    
@Func_1: getCountrywiseDF(): Calls the loadJSONData function for the country specified and
 returns a dataframe with the complete data for one country.
 
@Func_2: loadJSONData(): Forms the appropriate URL using the base URL, country code and indicator code
 and sends the request to the endpoint. It returns a list of list of values for all features.

@return: The API returns the indicator values from the most recent year. Hence, creates a list of years in reverse order

"""

# Function to load JSON data from endpoint
def loadJSONData(country_code): 
    data_list = []
    
    # iterating over each indicator code specified in INDICATOR_CODES defined above
    for indicator in indic_codes: 
        
        # To Form the URL in desired format
        # E.g: http://api.worldbank.org/v2/countries/us/indicators/SP.POP.TOTL?format=json&per_page=200&date=1960:2018
        url = base_url + 'countries/' + country_code.lower() + '/indicators/' + indicator
        
        # sending the request using the resquests module
        rspns = requests.get(url, params=params)
        
        # validating the response status code
        # I'm checking if message is not present in the response as the API returns a status_code 200 even for error messages,
        # here, the response body contains a field called "message" which includes error details
        
        if rspns.status_code == 200 and ("message" not in rspns.json()[0].keys()):
            # print("Successfully got data for: " + str(featureMap[indicator]))
            
            # Values list for one feature
            indicator_vals = []
            
            # Here the response is an array which contians two arrays - [[{page: 1, ...}], [{year: 2018, SP.POP.TOTL: 123455}, ...]]
            # To check if the length of the response > 1
            if len(rspns.json()) > 1:
                
                # if yes, in response iterate over each object
                # For each object one single value is given for each year
                for obj in rspns.json()[1]:
                    
                    # checking if there is any empty values
                    if obj['value'] is "" or obj['value'] is None:
                        indicator_vals.append(None)
                    else:
                    # if a value is present, add it to the list of indicator values
                        indicator_vals.append(float(obj['value']))
                data_list.append(indicator_vals)
        else:
            # prints error message if calling API is failed
            print("Error in Loading the data. Status Code: " + str(rspns.status_code))
            
    # After all features have been obtained, I have added the values for the "Year"
    # This API returns the indicator values from the most recent year. Hence, I created a list of years in reverse order
    data_list.append([year for year in range(2018, 1959, -1)])
    # Below return returns the list of feature values [[val1,val2,val3...], [val1,val2,val3...], [val1,val2,val3...], ...]
    return data_list

#----------------------------------------------------------------------------------------------------

# Function invokes the loadJSONData and forms the final DataFrame for every country
def getCountrywiseDF(country_code):
    
    # Dataframe to have meaningful column names
    # Creating a list of column names from the map defined above
    col_list = list(feature_map.values())
    # Appending the column name: year
    col_list.append('Year')
    
    print("------------------Loading data for: " + country_map[country_code]+"-----------------------")
    
    # Calling the loadJSONData function and fetching the data from the API for the given country 
    data_list = loadJSONData(country_code)
    
    # transforming the lists of features into a DataFrame
    # Used np.column_stack to add each list as a column 
    data_f = pd.DataFrame(np.column_stack(data_list), columns = col_list)
    
    # using the country code, I added the country column by extracting the country name from the map 
    data_f['Country'] = country_map[country_code]
    
    # To display the resulting dataframe
    display(data_f.head())
    
    # for the given country it returns the formed dataframe 
    return data_f


# with the code of each country under consideration calling the getCountrywiseDF function 
# I got 7 seperate dataframes for each country

df_us = getCountrywiseDF('US')
df_in = getCountrywiseDF('IN')
df_cn = getCountrywiseDF('CN')
df_jp = getCountrywiseDF('JP')
df_ca = getCountrywiseDF('CA')
df_gb = getCountrywiseDF('GB')
df_za = getCountrywiseDF('ZA')

print("Complete Data is Loaded")

# Data Pre-processing -----------------------------------------------------------------------------------------

"""
The above data has just been collected through multiple API calls and combined together and
 dataframes contain some missing values in some of the features which means that the data
 needs some processing before using it for analysis

To ease the task of pre-processing and avoiding manually passing the dataframes
 Creating a list of the dataframes,and the copy() method has been used to avoid changing
 the original unprocessed dataframes. 

"""

# storing all the DataFrames in a list to apply pre-processing steps iteratively
lst_df = [df_us.copy(), df_in.copy(), df_cn.copy(), df_jp.copy(), 
           df_ca.copy(), df_gb.copy(), df_za.copy()]

"""
Dropping the features of majority values missing:
Identifying the features for missing values in a large number. Those aren't useful for analysis 
and removed from the dataset. 

The following function has been implemented to perform

Func: remove_missing_features(): 
    The above function takes a dataframe as a parameter to find that contain non-zero missing value features,
    and then finds the percentage of missing values in each of the columns. 
 
    The percentage of missing values is checked to drop the column and to return updated datframe when it is greater than 75%,
    But If a column has more than 75% missing values it is absolutely not useful for analysis.
This function is iteratively called on all of the dataframes created above.
 
"""

# Function identifies missing features and removes features that aren't useful

def remove_missing_features(df):
    
    # Ddataframe validation
    if df is None:
        print("No DataFrame received!")
        return
    
    # Dataframe copy to avoid changes to the original one
    df_copy = df.copy()
    
    print("Removing missing features for: " + df_copy.iloc[0]['Country'])
    
    # To find non-zero missing values features
    null_missing_vals = df.isnull().sum()

    # getting the index list with non-zero missing values features
    null_missing_index_list = list(null_missing_vals.index)
    
    # % of missing values is calculated
    # shape[0] gives the number of rows in the dataframe, So dividing the no. of missing values by total
    # no. of rows to get the ratio of missing values - multipled by 100 to get %
    # column name: percentage of missing values.Here, missing_percentage consists of key value pairs
    missing_perctg = null_missing_vals[null_missing_vals!=0]/df.shape[0]*100
    
    # listing for columns to drop
    cols_trim = []
    
    # iterating over each key value pair
    for i,val in enumerate(missing_perctg):
        # if percentage value is > 75
        if val > 75:
            # add the corresponding column to the list of cols_to_trim
            cols_trim.append(null_missing_index_list[i])

    # Example: cols_trim = ['Male Population']

    if len(cols_trim) > 0:
        # Using drop() method to drop the columns identified
        df_copy = df_copy.drop(columns = cols_trim)
        print("Dropped Columns:" + str(cols_trim))
    else:
        print("No columns dropped")

    # Updated dataframe is returned
    return df_copy

# To call the function on each DF for each country.
# on each dataframe in list_df through the map function in python, the function remove_missing_features will be applied 
lst_df = list(map(remove_missing_features, lst_df))

"""
Func:
    fill_missing_values(): 
        This fillna() function of pandas dataframes I used to fill values that are NaNs to fill missing values
    In observation from the raw dataframes populated above, the missing values are denoted by None.
    So, first filled NaN in place of None and then replaced the NaNs with the mean value of the columns.

"""

# Function which fills the rest over missing values with average values for columns

def fill_missing_values(df):
    
    # dataframes validation
    if df is None:
        print("No DataFrame received")
        return

    # created a copy
    df_cp1 = df.copy()
    
    print("Filling missing features for: "+df_cp1.iloc[0]['Country'])
    
    # dataframe to get the list of columns
    cols_list = list(df_cp1.columns)
    
    # excluding the last column - Country
    # explicitly  added column when data was loaded, so, it does not contain any missing values
    # performing an aggregation as fillna function doesn't work on categorical features
    cols_list.pop()
    
    # As fillna only works on nans, I replaced all None values with NaN
    df_cp1.fillna(value = pd.np.nan, inplace = True)
    
    # with the mean of the column values, I replaced all NaN values 
    for col in cols_list:
        df_cp1[col].fillna((df_cp1[col].mean()), inplace = True)

    print("Filling missing values completed")
    return df_cp1


# calling the function on each DF for each country.
# In list_df the map function, fill_missing_features will be applied on each dataframe 

lst_df = list(map(fill_missing_values, lst_df))


"""
For Categorical Features changing the type of Numeric:
As 'Year' column has numeric values and it's magnitude does not have a significance but it represents a period 
and not an actual number.So, I converted to a categorical variable.

Func:
    change_year_type(): This function is used to return the updated dataframe by taking a dataframe as a parameter
    and changes the dtype of year
"""

# Function which changes year type
def change_year_type(df):
    
    print("Changing type of Year for: "+df.loc[0]['Country'])
    # validation checks for existancy of year coloumn in dataframe
    if 'Year' in df.columns:
        # converts year to string
        df['Year'] = df.Year.astype(str)
    
    print("Completed changing type")
    # To return updated df
    return df

# calling function on each DF for each country.
# In list_df the map function fill_missing_features will be applied on each dataframe

lst_df = list(map(change_year_type, lst_df))

# Ddeleted features are dervied below

# India dataframe is picked from the list of dataframes
dfIndia = lst_df[1]

# Male Population = Total Population - Female Population (In any given year)
# validation for controlling any operation failures
if 'Total Population' in dfIndia.columns and 'Female Population' in dfIndia.columns:
    # calculating
    dfIndia["Male Population"] = dfIndia["Total Population"] - dfIndia["Female Population"]
else:
    # printing the error message
    print("One of the columns Total Population or Female Population is missing.")

lst_df[1].head()


# To check the no. of features
# Every DFe has the same number of columns. So, I have checked the first dataframe in the list

print('Total number of features: %d\n'%(lst_df[0].shape[1]))
lst_df[0].dtypes

"""
Store the cleaned dataset into CSV files:
 The dataframes for the 7 countries are stored into different CSV files for further analysis and the
 following function has been implemented for this task:

Func: write_data(): Iterates over the list of dataframes and writes them to CSV files with the name
 of the country. For example, the file containing data for India is stored in the file India.csv
"""

# Function conerts processed data to CSV files
def prcs_write_data():
    # validation checks to verify no. of countries and dataframes match
    assert len(list(country_map.keys())) == len(lst_df)
    
    # iterate over country names from the country map and the list of dataframes simultaneously
    for country_name, df_data in zip(countries, lst_df):
        print("Writing data for: " + country_name)
        file_name = country_name+".csv"
        # converting to CSV
        try:
            df_data.to_csv(file_name, index = False)
            print("Successfully created: " + file_name)
        except:
            # For any I/O error occurs
            print("Error in writing to: " + file_name)
        
# calling the function
prcs_write_data()


# Analysing and Summarising the cleaned dataset
# I have used the matplotlib and seaborn libraries to analyse the cleaned data to identify patterns and visualise.

"""
Reading the cleaned data from the CSV files:
Now, I'm using the cleaned dataset for analysis. First reading the CSV files I created previously.
"""
# reading the cleaned data from every CSV
try:
    us_cleaned_df = pd.read_csv('USA.csv')
    in_cleaned_df = pd.read_csv('India.csv')
    cn_cleaned_df = pd.read_csv('China.csv')
    jp_cleaned_df = pd.read_csv('Japan.csv')
    ca_cleaned_df = pd.read_csv('Canada.csv')
    gb_cleaned_df = pd.read_csv('Great Britain.csv')
    za_cleaned_df = pd.read_csv('South Africa.csv')
    print("Read all files Successfully")
except:
    # handling the I/O exceptions
    print("Error in reading a file. Please Check the file path and if a file exists with the name given.")

# display data of one country to check if the cleaned data is loaded
print("Displaying data for USA: ")
display(us_cleaned_df.head())

# For further analysis creating a list of clean dataframes 
cleaned_list_df = [us_cleaned_df, in_cleaned_df, cn_cleaned_df, jp_cleaned_df,
                   ca_cleaned_df, gb_cleaned_df, za_cleaned_df]

# prepared a combined DataFrame
combined_df = pd.concat(cleaned_list_df,sort = False)
combined_df.head(200)


"""
  Excluding the categorical features Year and Country from the combined DataFrame as descriptive statistics
  show the characteristics of numerical features and gives information such as count, mean, mininum and maximum values etc.
"""
# creating a copy to not disturb original DF
# year and country columns dropped
df_copy = combined_df.drop(['Year', 'Country'], axis='columns')
df_copy.describe()


#########_________Analysis Using Visualisation Plots_________##########

"""
    *** Comparing Population of Countries in 1960, 1980 and 2000, 2018: ***
    
Comparing how the population for different countries has changed in few years and following code below
prepares few DataFrames - consideration of each year. So extracting the column Total Population. 
The difference in population has been shown using a grouped bar chart.
 
"""

# GRAPH 1: ----------------------------------- 1960 AND 1980


# referring to list of countries
list_countries = countries
# intialise two dataframes df_1970, year 1990
df_1970 = pd.DataFrame()
df_1990 = pd.DataFrame()

# for each dataframe in the list of cleaned dataframes
for i, df in enumerate(cleaned_list_df):
    # pick the value of Total Population for year 2000 and 2018
    df_1970[list_countries[i]] = df[df['Year'] == 1970]["Total Population"]
    df_1990[list_countries[i]] = df[df['Year'] == 1990]["Total Population"]

# Resulting dataframes has countries in columns and two rows each for 1970 & 1990
# To draw a grouped bar plot I need years as columns, hence we take a transpose
df_1970 = df_1970.T
df_1990 = df_1990.T

# setting other global format
pd.options.display.float_format = '{:,.1f}'.format  

# Renaming columns to the year
df_1970 = df_1970.rename(columns={48 : 1970})
df_1990 = df_1990.rename(columns={28 : 1990})

# I joined the 2 dataframes for both years
df_2years= df_1970.join(df_1990)

# Here Country name is an Index, so added it as a column into data frame.
df_2years['Countries'] = df_2years.index

# Dropping the original index
df_2years.reset_index(drop = True)

print("Data of Total Population for 1970 and 1990 for all countries: ")
display(df_2years)

font1 = {'family':'serif','color':'blue','size':20}
font2 = {'family':'serif','color':'darkred','size':15}

plt.figure(figsize=(7, 5))
# plotting the Graph using the library matplotlib.pyplot
df_2years.plot(kind='bar', x='Countries', y=[1970, 1990])
plt.title("Total Population-Yearly", fontdict = font1)
plt.xlabel("Countries", fontdict = font2)
plt.ylabel("Total Population - Yearly Hike", fontdict = font2)


# GRAPH 2: ----------------------------------- 2000 AND 2018

# referring to list of countries
list_countries = countries
# intialise two dataframes df_2000 year 2018
df_2000 = pd.DataFrame()
df_2018 = pd.DataFrame()

# for each dataframe in the list of cleaned dataframes
for i,df in enumerate(cleaned_list_df):
    # pick the value of Total Population for year 2000 and 2018
    df_2000[list_countries[i]] = df[df['Year'] == 2000]["Total Population"]
    df_2018[list_countries[i]] = df[df['Year'] == 2018]["Total Population"]

# Resulting dataframes has countries in columns and two rows each for 2000 & 2018
# To draw a grouped bar plot I need years as columns, hence we take a transpose
df_2000 = df_2000.T
df_2018 = df_2018.T

# setting other global format
pd.options.display.float_format = '{:,.1f}'.format

# Renaming columns to the year
df_2000 = df_2000.rename(columns={18 : 2000})
df_2018 = df_2018.rename(columns={0 : 2018})

# I joined the 2 dataframes for both years
df_2years= df_2000.join(df_2018)

# Here Country name is an Index, so added it as a column into data frame.
df_2years['Countries'] = df_2years.index

# Dropping the original index
df_2years.reset_index(drop = True)

print("Data of Total Population for 2000 and 2018 for all countries: ")
display(df_2years)


font1 = {'family':'serif','color':'blue','size':20}
font2 = {'family':'serif','color':'darkred','size':15}

plt.figure(figsize=(7, 5))
# plotting the Graph using the library matplotlib.pyplot
df_2years.plot(kind='bar', x='Countries', y=[2000, 2018])
plt.title("Total Population-Yearly", fontdict = font1)
plt.xlabel("Countries", fontdict = font2)
plt.ylabel("Total Population -Yearly Hike", fontdict = font2)

# GRAPH 3: ----------------------------------------------------------------------------------------------------

"""
Population vs Electric Power Consumption for India and China:
    
As seen from the population grouped bar plot above, 
India and China are the most populated countries. So, here I examined the Electric power consumption 
in these countries. I have extracted the Country, Population, and Electric Power consumption 
from the dataframes for these 2 countries and visualised them using a scatter plot.

"""
# function whihc extracts particular columns from the DFs for India and China
def in_cn_form_df():
    # Data Frame for India
    in_df1 = in_cleaned_df[['Total Population', 'Electric Power Consumption(kWH per capita)', 'Country']]
    # Data Frame for China
    cn_df1 = cn_cleaned_df[['Total Population', 'Electric Power Consumption(kWH per capita)', 'Country']]
    # combining the 2 dataframes
    in_cn_df = pd.concat([in_df1, cn_df1])
    return in_cn_df

# I got the desired data
in_cn_df = in_cn_form_df()
print("Few records from the selected features: ")
display(in_cn_df.head())

# scatter plot with ease of seaborn library
plt.figure(figsize = (7, 5))
sns.set(style = "whitegrid")
ax=sns.scatterplot(x = 'Total Population', y = 'Electric Power Consumption(kWH per capita)', 
                   hue = 'Country', palette = "bright", data = in_cn_df).set_title('Electricity Consumption based on Population Growth')




# GRAPH 4: ----------------------------------------------------------------------------------------------------

"""
GDP for all countries of 6 years:
    
I have used a line graph to show how the GDP for various countries has varied over 6 years.
 By extracting the columns Year, GDP in USD and Country from each of the country's dataframe and 
 stored it in a smaller dataframe for plotting. The line plot clearly shows the trends over a period of time and 
 can also be used to compare the trends of different categories.
 
"""

# function below to form dataframe with GDP, Country andYear
def extracted_columns(df_cleaned):
    df = pd.DataFrame()
    # pick data sorted in descending for 6 years
    df['Year'] = df_cleaned.loc[:10, 'Year']
    df['GDP in USD'] = df_cleaned.loc[:10, 'GDP in USD']
    df['Country'] = df_cleaned.loc[:10, 'Country']
    return df

# function below fetches one dataframe with 3 features from each country
def gdp__form_df():
    # function call to extract_columns()
    in_df = extracted_columns(in_cleaned_df)
    us_df = extracted_columns(us_cleaned_df)
    cn_df = extracted_columns(cn_cleaned_df)
    jp_df = extracted_columns(jp_cleaned_df)
    ca_df = extracted_columns(ca_cleaned_df)
    gb_df = extracted_columns(gb_cleaned_df)
    za_df = extracted_columns(za_cleaned_df)
    
    # combining the 7 dfs into a single df with 3 columns
    # ignored the original index
    gdp_df = pd.concat([in_df, us_df, cn_df, jp_df, ca_df, gb_df, za_df], ignore_index = True)
    return gdp_df

# getting the combined DF
gdp_df = gdp__form_df()

print("Few records from the Dataframe containing Year, GDP and Country:")
display(gdp_df.head())

# setting figure size
plt.figure(figsize = (7, 5))
sns.set(style="whitegrid")

# plotting with ease of seaborn library
ax = sns.lineplot(x = 'Year', y = 'GDP in USD', hue = 'Country',
                style = "Country",palette = "Set2", markers = True,
                dashes = False, data = gdp_df, linewidth = 2.5).set_title('Country wise - GDP Growth for 6 Years')


# GRAPH 5: ----------------------------------------------------------------------------------------------------

"""
Variation in different Energy Consumption over the years for India:
    
For this analysis, I have chosen the energy consumption data for India over the years upto 2010 and
 plotted a multi-line chart to observe the trend.
"""

# Pick the columns Year, and 3 different power consumptions from the dataframe for russia
plt.plot(in_cleaned_df.loc[5:, ['Year']], in_cleaned_df.loc[5:, ['Electric Power Consumption(kWH per capita)']],'.-')
plt.plot(in_cleaned_df.loc[5:, ['Year']], in_cleaned_df.loc[5:, ['Renewable Energy Consumption (%)']],'.-')
plt.plot(in_cleaned_df.loc[5:, ['Year']], in_cleaned_df.loc[5:, ['Fossil Fuel Consumption (%)']],'.-')

plt.legend(['Electric Power Consumption(kWH per capita)',
            'Renewable Energy Consumption(%)', 'Fossil Fuel Consumption(%)'], loc = 'best')
plt.title("Different Energy Consumptions in India\n")
plt.xlabel('Year')
plt.ylabel('Energy Consumption')
plt.show()



# GRAPH 6 ------------------------------------------------------------------------------------------

"""
Variation in different Energy Consumption over the years for South Africa:
    
For this analysis, I have chosen the energy consumption data for South Africa over 6 years and
 plotted a multi-line chart to observe the trend.
"""

plt.plot(za_cleaned_df.loc[5:, ['Year']], za_cleaned_df.loc[5:, ['Electric Power Consumption(kWH per capita)']],'.-')
plt.plot(za_cleaned_df.loc[5:, ['Year']], za_cleaned_df.loc[5:, ['Renewable Energy Consumption (%)']],'.-')
plt.plot(za_cleaned_df.loc[5:, ['Year']], za_cleaned_df.loc[5:, ['Fossil Fuel Consumption (%)']],'.-')

plt.legend(['Electric Power Consumption(kWH per capita)',
            'Renewable Energy Consumption(%)', 'Fossil Fuel Consumption(%)'], loc = 'best')
plt.title("Different Energy Consumptions in South Africa\n")
plt.xlabel('Year')
plt.ylabel('Energy Consumption')
plt.show()

# GRAPH 7 ------------------------------------------------------------------------------------------


"""
Agriculture Value added to GDP for all countries of 6 years:
    
I have used a line graph to show how the Agriculture Value added to GDP for various countries has varied over 6 years.
 By extracting the columns Year, Agriculture Value added to GDP and Country from each of the country's dataframe and 
 stored it in a smaller dataframe for plotting. The line plot clearly shows the trends over a period of time and 
 can also be used to compare the trends of different categories.
 
"""

# function to to form a dataframe with Year, GDP and Country
def extracted_columns(df_cleaned):
    df = pd.DataFrame()
    # pick data for the recent 10 years, note that the data sorted in descending order of year
    df['Year'] = df_cleaned.loc[:10, 'Year']
    df['Agriculture value added(in USD)'] = df_cleaned.loc[:10, 'Agriculture value added(in USD)']
    df['Country']=df_cleaned.loc[:10, 'Country']
    return df

# function to fetch a single dataframe with 3 features from each country
def AgrVal_add_form_df():
    # function call to extract_columns()
    in_df = extracted_columns(in_cleaned_df)
    us_df = extracted_columns(us_cleaned_df)
    cn_df = extracted_columns(cn_cleaned_df)
    jp_df = extracted_columns(jp_cleaned_df)
    ca_df = extracted_columns(ca_cleaned_df)
    gb_df = extracted_columns(gb_cleaned_df)
    za_df = extracted_columns(za_cleaned_df)
    # combine the 7 dfs into a single df with 3 columns
    # we ignore the original index
    agr_df = pd.concat([in_df, us_df, cn_df, jp_df, ca_df, gb_df, za_df], ignore_index=True)
    return agr_df

# get the combined DF
agr_df = AgrVal_add_form_df()

print("Few records from the Dataframe containing Year, Agriculture value added to GDP and Country:")
display(agr_df.head())

# set figure size
plt.figure(figsize = (7, 5))
sns.set(style="whitegrid")

# plot using seaborn library
ax = sns.lineplot(x = 'Year', y = 'Agriculture value added(in USD)', hue = 'Country',
                style = "Country",palette = "Set2", markers = True,
                dashes = False, data = agr_df, linewidth = 2.5).set_title('Country wise - Agriculture Value added to GDP Growth for 6 Years')

# GRAPH 8 --------------------------------------------------------------------------------------------------

"""
Female Employement value added to GDP for all countries of 6 years:
    
I have used a line graph to show how the Female Employement value added to GDP for various countries has varied over 6 years.
 By extracting the columns Year, Female Employement value added to GDP and Country from each of the country's dataframe and 
 stored it in a smaller dataframe for plotting. The line plot clearly shows the trends over a period of time and 
 can also be used to compare the trends of different categories.
 
"""

def extracted_columns(df_cleaned):
    df = pd.DataFrame()
    # pick data sorted in descending for 6 years
    df['Year'] = df_cleaned.loc[:10, 'Year']
    df['Female Employment in Industry(%)'] = df_cleaned.loc[:10, 'Female Employment in Industry(%)']
    df['Country'] = df_cleaned.loc[:10, 'Country']
    return df

# function below fetches one dataframe with 3 features from each country
def fei__form_df():
    # function call to extract_columns()
    in_df = extracted_columns(in_cleaned_df)
    us_df = extracted_columns(us_cleaned_df)
    cn_df = extracted_columns(cn_cleaned_df)
    jp_df = extracted_columns(jp_cleaned_df)
    ca_df = extracted_columns(ca_cleaned_df)
    gb_df = extracted_columns(gb_cleaned_df)
    za_df = extracted_columns(za_cleaned_df)
    
    # combining the 7 dfs into a single df with 3 columns
    # ignored the original index
    fei_df = pd.concat([in_df, us_df, cn_df, jp_df, ca_df, gb_df, za_df], ignore_index = True)
    return fei_df

# getting the combined DF
fei_df = fei__form_df()

print("Few records from the Dataframe containing Year, Female employement in Industry and Country:")
display(fei_df.head())

 
#call the tabulate function to see the table
print(tabulate(fei_df, headers = 'keys', tablefmt = 'fancy_grid'))


