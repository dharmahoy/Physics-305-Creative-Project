'''
Dharma Hoy
04/23/22
Physics 305 Creative Project
Analyze data

This program takes a look at the original not normalized, 
data set that only includes the columns that will be used.
It creates graphs and finds the summary statistics 
that do a good job of showing what is contained in this
data set.
'''
import pandas as pd
import matplotlib.pyplot as plt

# load data
stars = pd.read_csv("data/smaller_star_classification.csv")

# summary statistics
stars.describe(include = 'all').to_csv("output/summaryStatistics.csv")

# take a closer look at the output class
classes = stars.groupby('class').size()
print(classes)

# output is shown below
'''
class
GALAXY    59445
QSO       18961
STAR      21594
dtype: int64
'''

# create a bar chart of the different classes
fig, ax = plt.subplots()
plt.bar(['Galaxy', 'Qasar', 'Star'], [59445, 18961, 21594], color = 'dodgerblue')
plt.title("Counts of Each Class")
plt.xlabel("Class")
plt.ylabel("Count")
plt.savefig("output/classCounts.pdf")
plt.show()
plt.close()

# create a scatter plot of the class vs redshift
'''
this plot has three lines instead of individual points
because there is so much data in this dataset. All of
the data is plotted as opposed to a sample to make
sure the entire range of data is shown for each class.
'''
fig, ax = plt.subplots()
plt.scatter(stars['class'], stars['redshift'], color = 'firebrick')
plt.title("Range of Redshift for Each Class")
plt.xlabel("Class")
plt.ylabel("Redshift Value")
plt.savefig("output/classRedshift.pdf")
plt.show()
plt.close()

# Create a histogram of the filter in the photometric system for
# all the wavelengths this is measured for. These colors are
# ultraviolet, green, red, near infrared, and infrared
fig, ax = plt.subplots(3,2)

ax[0,0].hist(stars['u'], bins = 20, range = (13,25), color = 'violet') 
ax[0,0].set(title = "Ultraviolet", xlabel = "Filter in Photometric System", ylabel = "Count")

ax[0,1].hist(stars['g'], bins = 20, range = (13,25), color = 'seagreen')
ax[0,1].set(title = "Green", xlabel = "Filter in Photometric System", ylabel = "Count")

ax[1,0].hist(stars['r'], bins = 20, range = (13,25), color = 'tomato')
ax[1,0].set(title = "Red", xlabel = "Filter in Photometric System", ylabel = "Count")

ax[1,1].hist(stars['i'], bins = 20, range = (13,25), color = 'mediumturquoise')
ax[1,1].set(title = 'Near Infared', xlabel = "Filter in Photometric System", ylabel = "Count")

ax[2,0].hist(stars['z'], bins = 20, range = (13,25), color = 'silver')
ax[2,0].set(title = "Infared", xlabel = "Filter in Photometric System", ylabel = "Count")

ax[2,1].set_visible(False)

plt.tight_layout()
plt.savefig("output/filterInPhotometricSystem.pdf")
plt.show()
plt.close()

# Create a histogram showing the distribution of the Modified Julian Date
fig, ax = plt.subplots()
plt.hist(stars['MJD'], bins = 15, color = 'lightgray')
plt.title("Distribution of Modified Julian Date")
plt.xlabel("Modified Julian Date")
plt.ylabel("Count")
plt.savefig("output/modifiedJulianDate.pdf")
plt.show()
plt.close()
