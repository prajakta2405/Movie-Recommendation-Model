'''
> DSBDA Mini Project <

##Topic - Movie Recommendation Model using scikit learn python module

Team Members - 
    - Ajay
    - Vivek
    - Prajakta
    - Shivani
    - Prasad
    - Nitesh

Categories : Holly/Bolly - Genre - Actors - Series
'''



#Importing ML Modules
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import art


#Reading The Dataset
df = pd.read_csv("movie_dataset.csv")
features = ['keywords','cast','genres','director']




#Returns features of a particular Movie row
def combFeatures(row):
    return row['keywords']+" "+row['cast']+" "+row['genres']+" "+row['director']

#Returns title from the index of the movie
def getTitle(index):
    return df[df.index == index]["title"].values[0]

#Returns index from the title of the movie and if not found returns -1
def getIndex(title):
    if(str((df[df.title == title])).startswith("Empty")):
        return -1
    return df[df.title == title]["index"].values[0]




#Removing Null/Empty Values from the data set and creating single feature column
for feature in features:
    df[feature] = df[feature].fillna('')
df["combinedFeatures"] = df.apply(combFeatures,axis=1)




#Plotting similarity using count matrix
cv = CountVectorizer()
countMatrix = cv.fit_transform(df["combinedFeatures"])
similarityElement = cosine_similarity(countMatrix)




#Introduction Art
print(art.text2art("Movie Recommender"))
print("\n==> This model asks for the movie of you liking and returns any amount of similar movies.\n")




#User Input
usersMovie = input("> Enter Movie Name : ")
movieIndex = getIndex(usersMovie)
if movieIndex == -1:
    while movieIndex == -1:
        print("[!] Movie Name Is Incomplete Or Movie Doesn't Exists In The Dataset, Please Try Again!\n")
        usersMovie = input("> Enter Movie Name : ")
        movieIndex = getIndex(usersMovie)
movieRetCount = int(input("> Enter No. Of Movies To Be Fetched : "))




#Fetching Similar Movies From The Dataset and Sorting
similarMovies = list(enumerate(similarityElement[movieIndex]))
sortedSimilar = sorted(similarMovies,key=lambda x:x[1],reverse=True)[1:]




#Printing The Results
print("\nTop "+ str(movieRetCount) +" Similar Movies Like "+ usersMovie +" Are: ")
i=0
for element in sortedSimilar:
    i=i+1
    print(str(i)+". "+getTitle(element[0]))
    if i == movieRetCount :
        break;
if i!=movieRetCount:
    print("Dataset Only Has "+ str(i) +" Movies Similar Like "+ usersMovie)
