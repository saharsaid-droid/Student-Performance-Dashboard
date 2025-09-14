import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

data = pd.read_csv("C:/Users/dell/Downloads/Student_performance_10k.csv")
df = pd.DataFrame(data)
#print(df.head())
print(df.info())
print(df.describe())

    # drop missing value from columns
df = df.dropna(subset=["roll_no"])
df = df.dropna(subset=["gender"])
df = df.dropna(subset=["parental_level_of_education"])
df = df.dropna(subset=["race_ethnicity"])
df = df.dropna(subset=["lunch"])
df = df.dropna(subset=["test_preparation_course"])
print(df.info())
    #convert to float and will convert invalid entries to NaN
df['math_score'] = pd.to_numeric(df['math_score'], errors='coerce') 

    # fill missing value in a numeric columns
df['total_score'] = df['total_score'].fillna(df['math_score'] + df['reading_score'] + df['writing_score'] + df['science_score'] )
df['math_score'] = df['math_score'].fillna(df['total_score'] - (df['reading_score'] + df['writing_score'] + df['science_score'] ))
df['reading_score'] = df['reading_score'].fillna(df['total_score'] -(df['math_score'] + df['writing_score'] + df['science_score']) )
df['writing_score'] = df['writing_score'].fillna(df['total_score'] - (df['reading_score'] + df['math_score'] + df['science_score'] ))
df['science_score'] = df['science_score'].fillna(df['total_score'] - (df['reading_score'] + df['math_score'] + df['writing_score'] ))
print(df.info())

    # drop missing value that i can't fill in numeric columns
df = df.dropna(subset=["math_score"])
df = df.dropna(subset=["reading_score"])
df = df.dropna(subset=["writing_score"])
df = df.dropna(subset=["science_score"])
df = df.dropna(subset=["total_score"])

    # fun to fill missing value in grade
def determine_grade(total_score):
        if total_score >= 320:
            return 'A'
        elif total_score >= 250:
            return 'B'
        elif total_score >= 200:
            return 'C'
        elif total_score >= 150:
            return 'D'
        else:
            return 'Fail'


df['grade'] = df['total_score'].apply(determine_grade)


df.info()
df.duplicated().any()


    # create box plot to show outliers 
columns = ['math_score','reading_score','writing_score','science_score', 'total_score']
for i, column in enumerate(columns, 1):
        plt.subplot(3, 2, i)
        sns.boxplot(data=df, y=column, color='#8E4585')  
        plt.title(f'Box Plot - {column}')
        plt.ylabel(column)

plt.tight_layout()
plt.show()

    # Function to remove outliers based on IQR
def remove_outliers(data, column):
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

    # Remove outliers only for 'science_score' , 'writing_score' and 'total_score'
df_filtered = df.copy()
for column in ['science_score', 'writing_score','total_score']:
        df_filtered = remove_outliers(df_filtered, column)


# Calculate mean and variance for the selected columns before removing outliers
mean_before_outliers = df[columns].mean()
variance_before_outliers = df[columns].var()
print(mean_before_outliers)
print(variance_before_outliers)

# Calculate mean and variance for the selected columns after removing outliers
mean_after_outliers = df_filtered[columns].mean()
variance_after_outliers = df_filtered[columns].var()
print(mean_after_outliers)
print(variance_after_outliers)
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Plot the mean scores before outliers
axs[0, 0].bar(mean_before_outliers.index, mean_before_outliers.values, color='#9B4F96') 
axs[0, 0].set_title("Mean of Scores Before Outliers")
axs[0, 0].set_ylabel("Score")

# Plot the mean scores after outliers
axs[0, 1].bar(mean_after_outliers.index, mean_after_outliers.values, color='#9B0088')  
axs[0, 1].set_title("Mean of Scores After Outliers")
axs[0, 1].set_ylabel("Score")

# Plot the variance scores before outliers
axs[1, 0].bar(variance_before_outliers.index, variance_before_outliers.values, color='#B0C4DE')  
axs[1, 0].set_title("Variance of Scores Before Outliers")
axs[1, 0].set_ylabel("Variance")

# Plot the variance scores after outliers
axs[1, 1].bar(variance_after_outliers.index, variance_after_outliers.values, color='#A1C6EA')  
axs[1, 1].set_title("Variance of Scores After Outliers")
axs[1, 1].set_ylabel("Variance")

# Adjust the layout for better spacing
plt.tight_layout()
plt.show()

    # compare between gender before and after removing outliers
    # Before removing outliers
group_counts_before = df['gender'].value_counts()
total_count_before = group_counts_before.sum()
group_percentages_before = (group_counts_before / total_count_before) * 100

    # After removing outliers
group_counts_after = df_filtered['gender'].value_counts()
total_count_after = group_counts_after.sum()
group_percentages_after = (group_counts_after / total_count_after) * 100
colors = [ '#DB7093', '#87CEFA']  

    # Custom autopct function to hide small percentages
def custom_autopct(pct):
        return ('%1.1f%%' % pct) if pct >= 1 else ''  # Hide percentages less than 1%


    # Creating subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Pie chart before removing outliers
axes[0].pie(
        group_percentages_before,
        labels=group_percentages_before.index,
        autopct=custom_autopct,
        colors=colors,
        startangle=90
    )
axes[0].set_title('Distribution of Male and Female\nBefore Removing Outliers')
axes[0].axis('equal')  # Equal aspect ratio ensures the pie is drawn as a circle

    # Pie chart after removing outliers
axes[1].pie(
        group_percentages_after,
        labels=group_percentages_after.index,
        autopct=custom_autopct,
        colors=colors,
        startangle=90
    )
axes[1].set_title('Distribution of Male and Female\nAfter Removing Outliers')
axes[1].axis('equal')  # Equal aspect ratio ensures the pie is drawn as a circle

    # Adjust layout and show the plot
plt.tight_layout()
plt.show()

    #-----------------------------------------------

    # Before removing outliers
group_counts_before = df['race_ethnicity'].value_counts()
total_count_before = group_counts_before.sum()
group_percentages_before = (group_counts_before / total_count_before) * 100

    # After removing outliers
group_counts_after = df_filtered['race_ethnicity'].value_counts()
total_count_after = group_counts_after.sum()
group_percentages_after = (group_counts_after / total_count_after) * 100

    # Colors for the pie chart
colors =['#FF6347', '#FFD700', '#98FB98', '#20B2AA', '#FF1493', '#8A2BE2']
 

    # Custom autopct function to hide 0.0%
def custom_autopct(pct):
        return ('%1.1f%%' % pct) if pct >= 1 else ''  # Hide percentages less than 1%

    # Create subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Pie chart for "before removing outliers"
axes[0].pie(
        group_percentages_before,
        labels=group_percentages_before.index,
        autopct=custom_autopct,
        colors=colors,
        startangle=90
    )
axes[0].set_title('Distribution of Groups\nBefore Removing Outliers')
axes[0].axis('equal')  # Ensure the pie is drawn as a circle

    # Pie chart for "after removing outliers"
axes[1].pie(
        group_percentages_after,
        labels=group_percentages_after.index,
        autopct=custom_autopct,
        colors=colors,
        startangle=90
    )
axes[1].set_title('Distribution of Groups\nAfter Removing Outliers')
axes[1].axis('equal')  # Ensure the pie is drawn as a circle

    # Adjust layout and display
plt.tight_layout()
plt.show()

    #--------------------------------

    # compare between grade before and after removing outliers
    # Before removing outliers
group_counts_before = df['grade'].value_counts()
total_count_before = group_counts_before.sum()
group_percentages_before = (group_counts_before / total_count_before) * 100

    # After removing outliers
group_counts_after = df_filtered['grade'].value_counts()
    # Remove "Fail" category from the "After Removing Outliers" data
group_counts_after = group_counts_after.drop(labels='Fail', errors='ignore')
total_count_after = group_counts_after.sum()
group_percentages_after = (group_counts_after / total_count_after) * 100

    # Colors for the pie chart
colors = ['#D76689', '#FF91A1', '#D53D6B', '#D39BC2', '#D6A1D3']  

    # Custom autopct function to hide 0.0% and ensure clarity for Fail
def custom_autopct(pct, all_vals):
        absolute = int(round(pct / 100. * sum(all_vals)))
        return f'{pct:.1f}%' if pct > 0 else ''


    # Ensure consistent labels and colors for "Before Removing Outliers"
unique_labels_before = group_counts_before.index

    # Pie chart for "Before Removing Outliers"
fig, axes = plt.subplots(1, 2, figsize=(14, 7))

axes[0].pie(
        group_percentages_before,
        labels=unique_labels_before,
        autopct=lambda pct: custom_autopct(pct, group_counts_before),
        colors=colors[:len(unique_labels_before)],
        startangle=90
    )
axes[0].set_title('Distribution of Grades\nBefore Removing Outliers')
axes[0].axis('equal')  # Ensure the pie is drawn as a circle

    # Pie chart for "After Removing Outliers" (without "Fail")
unique_labels_after = group_counts_after.index
axes[1].pie(
        group_percentages_after,
        labels=unique_labels_after,
        autopct=lambda pct: custom_autopct(pct, group_counts_after),
        colors=colors[:len(unique_labels_after)],
        startangle=90
    )
axes[1].set_title('Distribution of Grades\nAfter Removing Outliers')
axes[1].axis('equal')  # Ensure the pie is drawn as a circle

    # Adjust layout and display
plt.tight_layout()
plt.show()


    # Create  sub plots to compare between numeric column 
    # before and after removing outlier
fig, axes = plt.subplots(2, 2, figsize=(12, 10))  # Adjust figsize as needed

    # Plot each histogram in its respective subplot
axes[0, 0].hist(df['math_score'], color='skyblue', alpha=0.5, bins=10)
axes[0, 0].set_title('Distribution of Math Scores before removing outliers')
axes[0, 0].set_xlabel('Math Score')
axes[0, 0].set_ylabel('Frequency')

axes[0, 1].hist(df_filtered['math_score'], color='lightgreen', alpha=0.5, bins=10)
axes[0, 1].set_title('Distribution of Math Scores after removing outliers')
axes[0, 1].set_xlabel('math Score')
axes[0, 1].set_ylabel('Frequency')

axes[1, 0].hist(df_filtered['writing_score'], color='lightcoral', alpha=0.5, bins=10)
axes[1, 0].set_title('Distribution of Writing Scores before removing outliers')
axes[1, 0].set_xlabel('Writing Score')
axes[1, 0].set_ylabel('Frequency')

axes[1, 1].hist(df_filtered['writing_score'], color='gold', alpha=0.5, bins=10)
axes[1, 1].set_title('Distribution of Writing Scores after removing outliers')
axes[1, 1].set_xlabel('writing_score')
axes[1, 1].set_ylabel('Frequency')

    # Adjust layout to prevent overlapping titles and labels
plt.tight_layout()

    # Show the plot
plt.show()



fig, axes = plt.subplots(2, 2, figsize=(12, 10))  

    # Plot each histogram in its respective subplot
axes[0, 0].hist(df['reading_score'], color='skyblue', alpha=0.5, bins=10)
axes[0, 0].set_title('Distribution of Reading Scores before removing outliers')
axes[0, 0].set_xlabel('Reading Score')
axes[0, 0].set_ylabel('Frequency')

axes[0, 1].hist(df_filtered['reading_score'], color='lightgreen', alpha=0.5, bins=10)
axes[0, 1].set_title('Distribution of Reading Scores after removing outliers')
axes[0, 1].set_xlabel('Reading Score')
axes[0, 1].set_ylabel('Frequency')

axes[1, 0].hist(df['science_score'], color='lightcoral', alpha=0.5, bins=10)
axes[1, 0].set_title('Distribution of Science Scores before removing outliers ')
axes[1, 0].set_xlabel('science_score')
axes[1, 0].set_ylabel('Frequency')

axes[1, 1].hist(df_filtered['science_score'], color='gold', alpha=0.5, bins=10)
axes[1, 1].set_title('Distribution of Science Scores after removing outliers ')
axes[1, 1].set_xlabel('Science Score')
axes[1, 1].set_ylabel('Frequency')




    # Adjust layout to prevent overlapping titles and labels
plt.tight_layout()

    # Show the plot
plt.show()


fig, axes = plt.subplots(1, 2, figsize=(12, 10))  # 1 row, 2 columns

    # Plot the first histogram
axes[0].hist(df['total_score'], color='skyblue', alpha=0.5, bins=10)
axes[0].set_title('Distribution of Total Scores before removing outliers')
axes[0].set_xlabel('Total Score')
axes[0].set_ylabel('Frequency')

    # Plot the second histogram
axes[1].hist(df_filtered['total_score'], color='darkblue', alpha=0.5, bins=10)
axes[1].set_title('Distribution of Total Scores after removing outliers')
axes[1].set_xlabel('Total Score')
axes[1].set_ylabel('Frequency')

    # Adjust layout
plt.tight_layout()

    # Show the plot
plt.show()



