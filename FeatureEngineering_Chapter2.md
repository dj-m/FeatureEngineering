# Feature Engineering: Chapter 2 - Dealing with Messy Data #
Notes from the second chapter of the DataCamp Feature Engineering course accessible [here](https://learn.datacamp.com/courses/feature-engineering-for-machine-learning-in-python).

![slide 1](ch2slides/ch2_01.png)

- This lesson will explore the concept of messy and missing values, how to find them, and once identified how to deal with them.

| ![slide 2](ch2slides/ch2_02.png) |
| :-: |
| Real-world data often has noise or omissions that can stem from many sources, like those above. |

- Data collection issue:
  - Paper surveys not being filled out fully
- Collection & Management Errors:
  - Making a mistake in transcribing the data.
- Omission issue:
  - Respondents skipping the age box in an online form.
- Transformation issue:
  - Average of a field with missing data.
  
![slide 3](ch2slides/ch2_03.png)

- Many machine learning models can't work with missing values.
  - A linear regression models needs a value for every row and column used in the data set.
- Missing data can clue you in to a data pipeline issue.
  - If data is consistently missing in a certain column, you ought to investigate as to why this is the case.
- Missing data can also provide information in and of itself:
  - a missing value for children means no children.
  
| ![slide 4](ch2slides/ch2_04.png) |
| :-: |
| Use **info()** method for a preliminary look at data completeness. |

- Here you can see that StackOverflorJobsRecommend, Gender and RawSalary are highly underpopulated. Thi prompts investigating where these missing values occur.
- Thislist is useful but becomes limited with larger datasets that have values missing scattered all over their features.

![slide 5](ch2slides/ch2_05.png)

- To find where these missing values exist, use the **isnull()** method.
  - All cells where missing values exist are shown as True.
  
| ![slide 6](ch2slides/ch2_06.png) |
| :-: |
| Count the number of missing values in a column by chaining **isnull()** with **sum()** methods. |
| ![slide 7](ch2slides/ch2_07.png) |
| Using **notnull()** will show you non-missing values (False). |

- Usage of **isnull()** & **notnull()** is allowed on the DataDrame as a whole and on each individual column.

![slide 8](ch2slides/ch2_08.png)

- How sparse is my data?
  - Most data sets contain missing values, often represented as NaN (Not a Number). If you are working with Pandas you can easily check how many missing values exist in each column. <br><br> Let's find out how many of the developers taking the survey chose to enter their age (found in the Age column of so\_survey_df) and their gender (Gender column of so\_survey_df).
  
	    # Subset the DataFrame
	    sub_df = so_survey_df[['Age','Gender']]
        
	    # Print the number of non-missing values
	    print(sub_df.notnull().sum())

- Based on the results, the **Gender** column has 693 non-missing entries.

- Finding the missing values
  - While having a summary of how much of your data is missing can be useful, often you will need to find the exact locations of these missing values. Using the same subset of the StackOverflow data from the last exercise (sub_df), you will show how a value can be flagged as missing.
  
	    # Print the top 10 entries of the DataFrame
	    print(sub_df.head(10))
	
	    # Print the locations of the missing values
	    print(sub_df.head(10).isnull())
	
	    # Print the locations of the non-missing values
	    print(sub_df.head(10).notnull())

  - Finding where the missing values exist can often be important.
  
![slide 9](ch2slides/ch2_09.png)

- Now that we can find the missing data, let's learn how to deal with them.

| ![slide 10](ch2slides/ch2_10.png) |
| :-: |
| Under **listwise deletion** the first and third rows will be dropped because of missing values in the _ConvertedSalary_ column. |

- If you're confident the missing values are occuring randomly (not intentionally being omitted) the most effective & statistically sound approach to dealing with them is called **complete case analysis** or **listwise deletion**.
  - In this method, a record is fully excluded from your model if any of its values are missing.
  
![slide 11](ch2slides/ch2_11.png)

 - Using pandas to implement listwise deletion, set the _how_ argument to 'any', to delete all rows with at least one missing value.
 
| ![slide 12](ch2slides/ch2_12.png) |
| :-: |
| To drop rows with missing values in a specific column, use the _subset_ argument. |

- Pass a list of columns to the _subset_ argument to specify which columns to consider when deleting rows.

![slide 13](ch2slides/ch2_13.png)

- While the prefereable approach to missing data is listwise deletion, there're drawbacks:
  - Any valid data points that share a row with th emissing values ger deleted.
  - If values don't occur at random, it can negatively affect the model.
  - Removing a feature instead of a row reduces the degrees of freedom of your model.
  
![slide 14](ch2slides/ch2_14.png)

- The most common way to deal with missing values is to use the **fillna()** method.
  - You provide the value that you want to replace the missing values with.
  - With categorical columns, it's common to replace missing values with strings like 'Other', 'Not Given', etc.
  - To modify the values in-place, in the original DataFrame, set the _inplace_ argument to True.
  
![slide 15](ch2slides/ch2_15.png)

- When you believe that the absence or presence of data is more important than the values themselves, you can create a column that records the absence of data and then drop the original column.
  - Call the **notnull()** method on a specific column, recording with True or False the presence of data.
  - To drop columns from a DataFrame, you can use the **drop()** method.
    - Specify a list of column names you want to drop via the _columns_ argument.

![slide 16](ch2slides/ch2_16.png)

- Listwise deletion
  - The simplest way to deal with missing values in your dataset when they are occurring entirely at random is to remove those rows, also called 'listwise deletion'.<br><br> Depending on the use case, you will sometimes want to remove all missing values in your data while other times you may want to only remove a particular column if too many values are missing in that column.
  
		# Print the number of rows and columns
		print(so_survey_df.shape)
		
		# Create a new DataFrame dropping all incomplete rows
		no_missing_values_rows = so_survey_df.dropna()
		
		# Print the shape of the new DataFrame
		print(no_missing_values_rows.shape)
		
		# Create a new DataFrame dropping all columns with incomplete rows
		no_missing_values_cols = so_survey_df.dropna(how = 'any', axis=1)
		
		# Print the shape of the new DataFrame
		print(no_missing_values_cols.shape)
		
		# Drop all rows where Gender is missing
		no_gender = so_survey_df.dropna(subset = ['Gender'])
		
		# Print the shape of the new DataFrame
		print(no_gender.shape)

  -  As you can see dropping all rows that contain any missing values may greatly reduce the size of your dataset. So you need to think carefully and consider several trade-offs when deleting missing values.

- Replacing missing values with constants
  - While removing missing data entirely maybe a correct approach in many situations, this may result in a lot of information being omitted from your models.<br><br> You may find categorical columns where the missing value is a valid piece of information in itself, such as someone refusing to answer a question in a survey. In these cases, you can fill all missing values with a new category entirely, for example 'No response given'.
  
		# Print the count of occurrences
		print(so_survey_df['Gender'].value_counts())
		
		# Replace missing values
		so_survey_df['Gender'].fillna('Not Given', inplace = True)
		
		# Print the count of each value
		print(so_survey_df['Gender'].value_counts())

  - By filling in these missing values you can use the columns in your analyses.

![slide 17](ch2slides/ch2_17.png)

- Listwise Deletion will often not be feasible in real world use cases.

![slide 18](ch2slides/ch2_18.png)

- A common issue with removing all the rows with missing values is in building a predictive model.
  - When training your model, removing rows would quickly result in errors if your test data set had missing values, where you do not have the option of just not predicting these rows.

![slide 19](ch2slides/ch2_19.png)

- The alternative:
  - Replacing missing values.

![slide 20](ch2slides/ch2_20.png)

- What's a suitable value?
  - Using a commonly occuring value along the lines of a mean or median.
    - Using these methods can lead to biased estimates of the variances and covariances of the features.
	- The standard error and test statistics can be incorrectly estimated.
	- If these emtrics are needed, they should be calculated before the missing values have been filled.

| ![slide 21](ch2slides/ch2_21.png) |
| :-: |
| Calculate directly from a pandas Series by calling the mean/median method on the series. The missing values are excluded by default when calculating these statistics. |
| ![slide 22](ch2slides/ch2_22.png) |
| Fill the missing values, using **fillna()** with the **mean()** of that column. |

- Using **mean()** may result in too many decimal places.
  - Using the **astype()** method will remove all the decimal places.
  
| ![slide 23](ch2slides/ch2_23.png) |
| :-: |
| You're also able to round the mean before filling the missing values with it. |

![slide 24](ch2slides/ch2_24.png)

- Filling continuous missing values
  - In the last lesson, you dealt with different methods of removing data missing values and filling in missing values with a fixed string. These approaches are valid in many cases, particularly when dealing with categorical columns but have limited use when working with continuous values. In these cases, it may be most valid to fill the missing values in the column with a value calculated from the entries present in the column.
  
		# Print the first five rows of StackOverflowJobsRecommend column
		print(so_survey_df.StackOverflowJobsRecommend.head())
		
		# Fill missing values with the mean
		so_survey_df['StackOverflowJobsRecommend'].fillna(so_survey_df['StackOverflowJobsRecommend'].mean(), inplace = True)
		
		# Print the first five rows of StackOverflowJobsRecommend column
		print(so_survey_df['StackOverflowJobsRecommend'].head())
		
		# Fill missing values with the mean
		so_survey_df['StackOverflowJobsRecommend'].fillna(so_survey_df['StackOverflowJobsRecommend'].mean(), inplace=True)
		
		# Round the StackOverflowJobsRecommend values
		so_survey_df['StackOverflowJobsRecommend'] = round(so_survey_df['StackOverflowJobsRecommend'])
		
		# Print the top 5 rows
		print(so_survey_df['StackOverflowJobsRecommend'].head())
  - Remember you should only round your values if you are certain it is applicable.

| ![exercise 3 multiple choice](ch2slides/ex3_choice.png) |
| :-: |
| Never calculate values based on your test set. Values calculated on the train test should be applied to both DataFrames. |

![slide 25](ch2slides/ch2_25.png)

- You'll come across features that will need to be updated in some other ways than filling in their null/missing values.
  - Looking at monetary values:
    - If the dataset came from excel, it may contain currency signs or commas that prevent pandas form reading it as numeric values.

| ![slide 26](ch2slides/ch2_26.png) |
| :-: |
| Intuitively you know that RawSalary should be numeric, but why is that? |
| ![slide 27](ch2slides/ch2_27.png) |
| Numeric columns shouldn't contain non-numeric characters. |
| ![slide 28](ch2slides/ch2_28.png) |
| To remove these commas, use string (str) methods like **replace()** to remove all occurrences of comma. The first argument is what you want to replace with the second argument being what you want to replace it with. |

- To alter the data type of the column, use **astype()** method.
- If an error results form converting a column's data type, it indicates stray characters which you didn't account for.
  - Instead of manually searching for stray caharacters use **to\_numeric()** function along with the _errors_ argument.
    - errors = coerce means that values not able to be converted to numeric will be replaced with NaN.

| ![slide 29](ch2slides/ch2_29.png) |
| :-: |
| Use the **isna()** method to find which values failed to parse. It looks like dollar signs are in the data. Use the **replace()** method, as before, to remove the dollar signs.|

![slide 30](ch2slides/ch2_30.png)

- If you're applying different methods or the same one several times, you can chain the methods, calling one after the other to obtain the desirec result.
  - For example:
    - Cleaning up characters
	- Changing the data type
	- Normalizing the values

![slide 31](ch2slides/ch2_31.png)

- Dealing with stray characters (I)
  - In this exercise, you will work with the RawSalary column of so_survey_df which contains the wages of the respondents along with the currency symbols and commas, such as $42,000. When importing data from Microsoft Excel, more often that not you will come across data in this form.
  
		# Remove the commas in the column
		so_survey_df['RawSalary'] = so_survey_df['RawSalary'].str.replace(',', '')
		
		# Remove the dollar signs in the column
		so_survey_df['RawSalary'] = so_survey_df['RawSalary'].str.replace('$','')

  - Replacing/removing specific characters is a very useful skill.
  
- Dealing with stray characters (II)
  - In the last exercise, you could tell quickly based off of the df.head() call which characters were causing an issue. In many cases this will not be so apparent. There will often be values deep within a column that are preventing you from casting a column as a numeric type so that it can be used in a model or further feature engineering.<br><br> One approach to finding these values is to force the column to the data type desired using pd.to_numeric(), coercing any values causing issues to NaN, Then filtering the DataFrame by just the rows containing the NaN values.<br><br> Try to cast the RawSalary column as a float and it will fail as an additional character can now be found in it. Find the character and remove it so the column can be cast as a float.
  
		# Attempt to convert the column to numeric values
		numeric_vals = pd.to_numeric(so_survey_df['RawSalary'], errors='coerce')
		
		# Find the indexes of missing values
		idx = numeric_vals.isna()
		
		# Print the relevant rows
		print(so_survey_df['RawSalary'][idx])
		
		# Replace the offending characters
		so_survey_df['RawSalary'] = so_survey_df['RawSalary'].str.replace('£','')
		
		# Convert the column to float
		so_survey_df['RawSalary'] = so_survey_df['RawSalary'].astype('float')
		
		# Print the column
		print(so_survey_df['RawSalary'])

  - Remember that even after removing all the relevant characters, you still need to change the type of the column to numeric if you want to plot these continuous values.
  
- Method chaining
  - When applying multiple operations on the same column (like in the previous exercises), you made the changes in several steps, assigning the results back in each step. However, when applying multiple successive operations on the same column, you can "chain" these operations together for clarity and ease of management. This can be achieved by calling multiple methods sequentially:
  
		# Method chaining
		df['column'] = df['column'].method1().method2().method3()
		
		# Same as 
		df['column'] = df['column'].method1()
		df['column'] = df['column'].method2()
		df['column'] = df['column'].method3()

  - In this exercise you'll apply the same methods as before but chained.
  
		# Use method chaining
		so_survey_df['RawSalary'] = so_survey_df['RawSalary']\
		                              .str.replace(',','')\
		                              .str.replace('$','')\
		                              .str.replace('£','')\
		                              .astype('float')
									   
		# Print the RawSalary column
		print(so_survey_df['RawSalary'])

  - Custom functions can be also used when method chaining using the **.apply()** method.
