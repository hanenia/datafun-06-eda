Project 6 EBA Notebook
Author: Hanna Anenia
Date: 2/12/24
Title: Datafun-06-eda
Overview Project 6 to creatcustome exploratory data analysis(EDA)
1. Environment Setup
Create and activate the project virtual environment, creat READ.me fiel
 update or generate a requirements.txt file
   py -m venv .venv
   .venv\Scripts\Activate 
Create Virtual Environment
   Create .gitignore file
   ni .gitignore
Add requirements folder
  ni requirements.txt
   py -m pip install -r requirements.txt
 Add gitignore
   ni gitignore
2. Project Setup and Install
   py -m pip install jupyterlab
   py -m pip install numpy
   py -m pip install pandas
   py -m pip install pyarrow
   py -m pip install matplotlib 
   py -m pip install seaborn
   py -m pip install scipy
   
3. Freeze dependencies
   py -m pip freeze > requirements.txt
   Git add and commit
  git add .
  git commit -m "add .gitignore, cmds to readme"
  git push origin main
  ort dependencies
Start Project
1. Import Dependencies
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy
Step 1. Data Acquisition

Load the Planets dataset into DataFrame
df = sns.load_dataset('planets')

# Inspect first rows of the DataFrame
print(df.head())
            method  number  orbital_period   mass  distance  year
0  Radial Velocity       1         269.300   7.10     77.40  2006
1  Radial Velocity       1         874.774   2.21     56.95  2008
2  Radial Velocity       1         763.000   2.60     19.84  2011
3  Radial Velocity       1         326.030  19.40    110.62  2007
4  Radial Velocity       1         516.220  10.50    119.47  2009
Step 2. Initial Data Inspection

# Display the 15 first rows of the dataframe planets
print(df.head(15))
print(df.shape)
print(df.dtypes)
             method  number  orbital_period   mass  distance  year
0   Radial Velocity       1         269.300   7.10     77.40  2006
1   Radial Velocity       1         874.774   2.21     56.95  2008
2   Radial Velocity       1         763.000   2.60     19.84  2011
3   Radial Velocity       1         326.030  19.40    110.62  2007
4   Radial Velocity       1         516.220  10.50    119.47  2009
5   Radial Velocity       1         185.840   4.80     76.39  2008
6   Radial Velocity       1        1773.400   4.64     18.15  2002
7   Radial Velocity       1         798.500    NaN     21.41  1996
8   Radial Velocity       1         993.300  10.30     73.10  2008
9   Radial Velocity       2         452.800   1.99     74.79  2010
10  Radial Velocity       2         883.000   0.86     74.79  2010
11  Radial Velocity       1         335.100   9.88     39.43  2009
12  Radial Velocity       1         479.100   3.88     97.28  2008
13  Radial Velocity       3        1078.000   2.53     14.08  1996
14  Radial Velocity       3        2391.000   0.54     14.08  2001
(1035, 6)
method             object
number              int64
orbital_period    float64
mass              float64
distance          float64
year                int64
dtype: object
Step 3. Initial Descriptive Statistics

#Descriptive statistics for passengers column

print(df['distance'].describe())
count     808.000000
mean      264.069282
std       733.116493
min         1.350000
25%        32.560000
50%        55.250000
75%       178.500000
max      8500.000000
Name: distance, dtype: float64
Step 5. Initial Data Distribution for Categorical Columns

# Inspect histogram by numerical column
df['number'].hist()
plt.xlabel('number (1 = No, 2 = Yes)')
plt.ylabel('Frequency')
plt.title('Distribution of number')

# Inspect histograms for all numerical columns
df.hist()

# Function to help with not overlapping of subplots
plt.tight_layout()

# Show all plots
plt.show()


Step 6. Initial Data Transformation and Feature Engineering Use pandas and other tools to perform transformations. Transformation may include renaming columns, adding new columns, or transforming existing data for more in-depth analysis.

# Inspect value counts by categorical column
df['orbital_period'].value_counts()

# Inspect value counts for all categorical columns
for col in df.select_dtypes(include=['object', 'category']).columns:
    # Display count plot
    sns.countplot(x=col, data=df)
    plt.title(f'Distribution of {col}')
    plt.show()

# Show all plots
plt.show()

6.1 Rename at least one column.

import seaborn as sns

# Load the planets dataset from Seaborn
planets_data = sns.load_dataset("planets")

# Renaming the 'mass' column to 'weigth'
planets_data.rename(columns={'mass': 'weight'}, inplace=True)


# Display the modified dataset
print(planets_data.head())
            method  number  orbital_period  weight  distance  year
0  Radial Velocity       1         269.300    7.10     77.40  2006
1  Radial Velocity       1         874.774    2.21     56.95  2008
2  Radial Velocity       1         763.000    2.60     19.84  2011
3  Radial Velocity       1         326.030   19.40    110.62  2007
4  Radial Velocity       1         516.220   10.50    119.47  2009
6.2 Add at least one column.

import seaborn as sns

# Load the planets dataset from Seaborn
planets_data = sns.load_dataset("planets")

# Add a new column 'mass*distance' based on mass 
planets_data['MD'] = planets_data['mass'] * planets_data['distance']

# Display the modified dataset
print(planets_data.head())
            method  number  orbital_period   mass  distance  year         MD
0  Radial Velocity       1         269.300   7.10     77.40  2006   549.5400
1  Radial Velocity       1         874.774   2.21     56.95  2008   125.8595
2  Radial Velocity       1         763.000   2.60     19.84  2011    51.5840
3  Radial Velocity       1         326.030  19.40    110.62  2007  2146.0280
4  Radial Velocity       1         516.220  10.50    119.47  2009  1254.4350
## Initial Data Distribution for Numerical Columns

import seaborn as sns
import matplotlib.pyplot as plt

# Load the Planets dataset from seaborn
planets = sns.load_dataset('planets')

# Visualize the first few rows of the dataset
print(planets.head())

filtered_planets_data = planets_data[planets_data['method'] == 'Radial Velocity'] # filtering the method column to where value is 'Radial Velocity'

plt.figure(figsize=(8,6))

plt.hist(planets_data['distance'], color='blue', edgecolor='pink')
plt.xlabel('distance (miles)')
plt.ylabel('Frequency (log scale)')
plt.title('Distribution of distance of radial velocity')
plt.yscale('log') # making the Y axis a log value for distribution
plt.grid(False)

# Show the plot
plt.show()
            method  number  orbital_period   mass  distance  year
0  Radial Velocity       1         269.300   7.10     77.40  2006
1  Radial Velocity       1         874.774   2.21     56.95  2008
2  Radial Velocity       1         763.000   2.60     19.84  2011
3  Radial Velocity       1         326.030  19.40    110.62  2007
4  Radial Velocity       1         516.220  10.50    119.47  2009

## Data Distribution for passengers column

# Set the style of the plot
sns.set_style("whitegrid")

# Filter data for the years 1996 to 2005
df_filtered = df[df['year'].between(1996, 2005)]

# Create a line plot with years on the x-axis and passengers on the y-axis
plt.figure(figsize=(10,5))
sns.lineplot(data=df_filtered, x='year', y='distance')
<Axes: xlabel='year', ylabel='distance'>

