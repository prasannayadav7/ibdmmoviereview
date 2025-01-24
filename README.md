# Movie-Review-Sentiment-Analysis
## Problem Statement
In this, we have to predict the number of positive and negative reviews based on sentiments by using different classification models.
## Data
Each row of data represents a review for a movie and the other column contains the predefinedsentiment of the review.

review : Moview reviews, textual unstructured data,

sentiment : predefined sentiment
the dataset click [here](https://www.kaggle.com/datasets/harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows)
## Approach
*  importing Necessary Libraries
*  Loading Data
*  Cleaning Textual data Removing HTML Tag Removing Accented Characters Expanding Contractions Removing special characters Lemmatizing Text Removing Stopwords
*  Sentiment Analysis Analysis with AFINN Analysis with SentiWordNet
*  In order to measure the performance of the model, we will be using the classification report which includes the accuracy, precision, recall, and F1 score for the model.
  ##  Technology used
 ### Libraries:
* Install Libraries: numpy, pandas, nltk, sklearn, seaborn, wordcloud.
* Load Data: Import IMDb reviews with pandas.
* Preprocess Text: Clean data (remove HTML, stopwords, punctuation), tokenize, lemmatize.
* Visualize: Plot sentiment distribution (seaborn) and word clouds (WordCloud).
* Vectorize: Convert text to numeric form using TfidfVectorizer or CountVectorizer.
* Train Models: Use Logistic Regression, Naive Bayes, or SVM (sklearn).
* Evaluate: Use accuracy, precision, and F1-score metrics.
 ### Platforms:
* VS Code, jupyter
## Factors
* Easy Text Cleaning: Uses simple techniques like removing unwanted words, punctuation, and converting text to lowercase.
* Clear Visuals: Shows sentiment distribution and word clouds for easy understanding.
* Multiple Models: Compares different models like Logistic Regression, Naive Bayes, and SVM to find the best fit.
* Flexible Workflow: Easily customizable to try different approaches or models.
* Scalable: Can be used in real apps or websites for real-time analysis.
* Simple Evaluation: Provides clear performance metrics to understand how well the model works.
## Model Accuracy
#### Logistic regression 
![image alt](https://github.com/prasannayadav7/ibdmmoviereview/blob/2fb2b31f311b12b9ffa226e2139aa2076ba01e7e/lr%20model%20accuracy.png)
#### Stochastic gradient descent or Linear support vector machine 
![image alt](https://github.com/prasannayadav7/ibdmmoviereview/blob/2fb2b31f311b12b9ffa226e2139aa2076ba01e7e/sdgc(svm)model%20accuracy.png)
#### Multinomial Naive Bayes
![image alt](https://github.com/prasannayadav7/ibdmmoviereview/blob/2fb2b31f311b12b9ffa226e2139aa2076ba01e7e/MultinomialNB%20model%20accuray.png)

## conclusion
* The sentiment analysis on IMDb movie reviews demonstrates the effectiveness of machine learning models in accurately predicting audience sentiment.
* By leveraging advanced text preprocessing techniques, vectorization methods like TF-IDF, and models such as Logistic Regression, Multinomial Naive Bayes, and SVM, the project achieves high accuracy in sentiment classification. The detailed evaluation metrics, including precision and recall, highlight the reliability of these models in understanding and analyzing movie reviews. 
* This project not only provides accurate insights but also proves to be a scalable and practical solution for real-world applications in entertainment and research industries.
* We can observed that both logistic regression and multinomial naive bayes model performing well compared to linear support vector machines.
* Still we can improve the accuracy of the models by preprocessing data and by using lexicon models like Text blob.



