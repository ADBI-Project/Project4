# Project4
##Supervised Learning Techniques for Sentiment Analytics

In this project, you will perform sentiment analysis over IMDB movie reviews and Twitter data.
Your goal will be to classify tweets or movie reviews as either positive or negative. 
Towards this end, you will be given labeled training to build the model and labeled testing data to evaluate the model. 
For classification, you will experiment with l ogistic regression as well as a N aive Bayes classifier from python’s well-regarded 
machine learning package scikit-learn. As a point of reference, Stanfords Recursive Neural Network code produced an accuracy of 51.1% 
on the IMDB dataset and 59.4% on the Twitter data.

A major part of this project is the task of generating feature vectors for use in these classifiers.
You will explore two methods: (1) A more traditional NLP technique where the features are simply “important” words and the 
feature vectors are simple binary vectors and (2) the Doc2Vec technique where document vectors are learned via 
artificial neural networks (a summary can be found here).
