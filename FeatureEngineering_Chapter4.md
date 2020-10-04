# Feature Engineering: Chapter 4 - Intro to Text Encoding #
Notes from the fourth chapter of the DataCamp Feature Engineering course accessible [here](https://learn.datacamp.com/courses/feature-engineering-for-machine-learning-in-python).

![slide 1](ch4slides/ch4_01.png)

| ![slide 2](ch4slides/ch4_02.png) |
| :-: |
| This is an example of free text associated with a speech given by George Washington. |

- Data that is not in a predefined form is called unstructured data, and free text data is a good example of this.
- Before you can leverage text data ina  machine learning model you must first convert it into a series of columns of numbers or vectors.

| ![slide 3](ch4slides/ch4_03.png) |
| :-: |
| The 'text' column contains the body of the speech text. |

- Before any text analytics can be performed, the text data must be in a format that can be used.

| ![slide 4](ch4slides/ch4_04.png) |
| :-: |
| Here, using regular expressions will allow you to select non-letter characters. |

- Most bodies of text will have non letter characters such as punctuation, that will need ot be removed before analysis.
- Selecting all the letter characters and placing a caret infront negates this, making the selected characters non-letter and replace it with a white space.

| ![slide 5](ch4slides/ch4_05.png) |
| :-: |
| The text of the first speech before and after processing via .replace() & regular expressions.|
| ![slide 6](ch4slides/ch4_06.png) |
| Next standardize the remaining characters by making them all lowercase. |

- Standardizing the text, by making them all lowercase ensure that the same word with/out capitalization won't be counted as separate words.

| ![slide 7](ch4slides/ch4_07.png) |
| :-: |
| The len() method calculates the number of characters in each speech. |

- You can even create features based on the content of different texts, but often there is value in the fundamental characteristics of a passage, such as its length.

![slide 9](ch4slides/ch4_09.png)

- You may want to learn how many words are contained in the text, by splitting the speech based on the white spaces and then count how many words after the split.
- First, use the split() method then chain the len() method to count the number of words in each speech.

| ![slide 10](ch4slides/ch4_10.png) |
| :-: |
| Simply dividing the character count by the number of words gives you the average word length. |

- You can also get the average word length.

![slide 11](ch4slides/ch4_11.png)

- Unstructured text data cannot be directly used in most analyses. Multiple steps need to be taken to go from a long free form string to a set of numeric columns in the right format that can be ingested by a machine learning model. The first step of this process is to standardize the data and eliminate any characters that could cause problems later on in your analytic pipeline. <br><br> In this chapter you will be working with a new dataset containing the inaugural speeches of the presidents of the United States loaded as speech_df, with the speeches stored in the text column.

        # Print the first 5 rows of the text column
		print(speech_df.text.head())
		
		# Replace all non letter characters with a whitespace
		speech_df['text_clean'] = speech_df['text'].str.replace('[^a-zA-Z]', ' ')
		
		# Change to lower case
		speech_df['text_clean'] = speech_df['text_clean'].str.lower()
		
		# Print the first 5 rows of the text_clean column
		print(speech_df['text_clean'].head())

  - Now your text strings have been standardized and cleaned up. You can now use this new column (text_clean) to extract information about the speeches.

		# Find the length of each text
		speech_df['char_cnt'] = speech_df['text_clean'].str.len()
		
		# Count the number of words in each text
		speech_df['word_cnt'] = speech_df['text_clean'].str.split().str.len()
		
		# Find the average length of word
		speech_df['avg_word_length'] = speech_df.char_cnt / speech_df.word_cnt
		
		# Print the first 5 rows of these columns
		print(speech_df[['text_clean', 'char_cnt', 'word_cnt', 'avg_word_length']])

![slide 12](ch4slides/ch4_12.png)

- Once high level info has been recorded you can begin creating features based on the actual content of each text.

| ![slide 13](ch4slides/ch4_13.png) |
| :-: |
| The most common approach to creating features is to create a column for each word. |
| ![slide 14](ch4slides/ch4_14.png) |
| scikit-learn provides the functionality to count the number of words. |
| ![slide 15](ch4slides/ch4_15.png) |
| Specify arguments in order to limit the number of columns created for text. |

- Creating a column for every word will result in far too many values for analyses.
- Specifying **min_df()** dictates the minimum number of texts that a word must be contained in. If a float is given, the word must appear in at least that percent of documents.
  - This threshold eliminates words that occur so rarely that they would not be useful when generalizing to new texts.
- **max_df()** limits words to only ones that occur below a certain percentage of data.
  - This can be useful to remove words that occur too frequently to be of any value.

| ![slide 16](ch4slides/ch4_16.png) |
| :-: |
| You call the **fit()** method on the relevant column. |

- Once the vectorizer's been instatiated you can fit it to the data you want to create your features around.

![slide 17](ch4slides/ch4_17.png)

- Once the vectorizer's been fit, you can call the **transform()** method on the column you want to transform. 
- The output is a sparse array with a row for every text and a column for every word that's been counted.

| ![slide 18](ch4slides/ch4_18.png) |
| :-: |
| To transform to a non-sparse array use **toarray()** method. |
| ![slide 19](ch4slides/ch4_19.png) |
| The output is an array without the concept of column names. |

- To get the names of the features that have been generated, use the **get_feature_names()** on the vectorizer.
  - This returns a list of the features generated, in the same order that the columns of the converted array are in.
  
| ![slide 20](ch4slides/ch4_20.png) |
| :-: |
| You can perform bot a fit and transform in one method called **fit_transform()**. |

- While fitting and transforming separately can be useful, particularly when you need to transform a different data set than the one that you used to fit the vectorizer to, you can accomplish both steps at once using the **fit_transform()** method.

| ![slide 21](ch4slides/ch4_21.png) |
| :-: |
| **add_prefix()** allows for distiguishing columns in the DataFrame. |

- Now with the array containing the count values of each words of interest, and a way to get the feature names, you can combine them into a DataFrame like above.

![slide 22](ch4slides/ch4_22.png)

- You can combine this DataFrame to your original DataFrame so they can be used to generate future analytical models using pandas **concat()** method.

![slide 23](ch4slides/ch4_23.png)

- Once high level information has been recorded you can begin creating features based on the actual content of each text. One way to do this is to approach it in a similar way to how you worked with categorical variables in the earlier lessons.

  - For each unique word in the dataset a column is created.
  - For each entry, the number of times this word occurs is counted and the count value is entered into the respective column.

- These "count" columns can then be used to train machine learning models.

		# Import CountVectorizer
		from sklearn.feature_extraction.text import CountVectorizer
		
		# Instantiate CountVectorizer
		cv = CountVectorizer()
		
		# Fit the vectorizer
		cv.fit(speech_df['text_clean'])
		
		# Print feature names
		print(cv.get_feature_names())

- Once the vectorizer has been fit to the data, it can be used to transform the text to an array representing the word counts. This array will have a row per block of text and a column for each of the features generated by the vectorizer that you observed in the last exercise.

		# Apply the vectorizer
		cv_transformed = cv.transform(speech_df['text_clean'])
		
		# Print the full array
		cv_array = cv_transformed.toarray()
		print(cv_array)
		
		# Print the shape of cv_array
		print(cv_array.shape)

- The speeches have 9043 unique words, which is a lot! In the next exercise, you will see how to create a limited set of features.

- As you have seen, using the CountVectorizer with its default settings creates a feature for every single word in your corpus. This can create far too many features, often including ones that will provide very little analytical value. <br><br>For this purpose CountVectorizer has parameters that you can set to reduce the number of features:
  - min_df : Use only words that occur in more than this percentage of documents. This can be used to remove outlier words that will not generalize across texts.
  - max_df : Use only words that occur in less than this percentage of documents. This is useful to eliminate very common words that occur in every corpus without adding value such as "and" or "the".

		# Import CountVectorizer
		from sklearn.feature_extraction.text import CountVectorizer
		
		# Specify arguements to limit the number of features generated
		cv = CountVectorizer(min_df = 0.2, max_df = 0.8)
		
		# Fit, transform, and convert into array
		cv_transformed = cv.fit_transform(speech_df['text_clean'])
		cv_array = cv_transformed.toarray()
		
		# Print the array shape
		print(cv_array.shape)

- Did you notice that the number of features (unique words) greatly reduced from 9043 to 818?

- Now that you have generated these count based features in an array you will need to reformat them so that they can be combined with the rest of the dataset. This can be achieved by converting the array into a pandas DataFrame, with the feature names you found earlier as the column names, and then concatenate it with the original DataFrame. <br><br> The numpy array (cv_array) and the vectorizer (cv) you fit in the last exercise are available in your workspace.

		# Create a DataFrame with these features
		cv_df = pd.DataFrame(cv_array, 
		                     columns=cv.get_feature_names()).add_prefix('Counts_')
							 
		# Add the new columns to the original DataFrame
		speech_df_new = pd.concat([speech_df, cv_df], axis=1, sort=False)
		print(speech_df_new.head())

- With the new features combined with the orginial DataFrame they can be now used for ML models or analysis.