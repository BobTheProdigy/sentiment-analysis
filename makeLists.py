import csv
import time
import re

# Purpose: makes reviews and ratings lists from reading CSV, Split the dataset into training, validation, and testing sets
# Parameters: None
# Return values: review list (string), score list (int)
def makeLists():
    startTime = time.time()
    csv.field_size_limit(1048576)
    review = []
    score = []
    numPositive = 0
    numNegative = 0
    
    # Read data from CSV
        # IMDB_MovieDataset code
        # with open('IMDB_MovieDataset.csv', encoding='utf-8') as csvfile:
        # reader = csv.DictReader(csvfile)
        # for row in reader:
        #     content = row['review']
        #     # if not re.search("[a-zA-Z]+", content): # may remove too many reviews, excludes commas and periods, need to fix -- 1/27 WC
        #     review.append(content)  # Append review text
        #     # getting rating, string -> number, append
        #     # ML needs to compare integers, can't interperet string review ratings
        #     rating = row['sentiment']
        #     if rating == 'positive':
        #         numPositive += 1
        #         score.append(1.0)
        #     else:
        #         numNegative += 1
        #         score.append(0.0)
    with open('steamReviews.csv', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            content = row['content']
            # if not re.search("[a-zA-Z]+", content): # may remove too many reviews, excludes commas and periods, need to fix -- 1/27 WC
            review.append(content)  # Append review text
            
            # getting rating, string -> number, append
            # ML needs to compare integers, can't interperet string review ratings
            rating = row['is_positive']
            if rating == 'Positive':
                numPositive += 1
                score.append(1.0)
            else:
                numNegative += 1
                score.append(0.0)
    print("positive overall reviews:", numPositive)
    print("negative overall reviews:", numNegative)
    
    # Calculate split indices
    totalLength = len(review)
    train_end_idx = int(totalLength * 0.8)  # 80% for training
    val_end_idx = train_end_idx + int(totalLength * 0.1)  # Additional 10% for validation

    # Split data into training, validation, and testing sets
    trainReviews = list(review[:train_end_idx])
    valReviews = list(review[train_end_idx:val_end_idx])
    testReviews = list(review[val_end_idx:])
    
    trainScores = list(score[:train_end_idx])
    valScores = list(score[train_end_idx:val_end_idx])
    testScores = list(score[val_end_idx:])


    print("traningList, validationList, testingList runtime: " + str(round(time.time() - startTime, 2)))
    return trainReviews, valReviews, testReviews, trainScores, valScores, testScores