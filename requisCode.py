import csv
import numpy as np
import torch
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import time
import matplotlib.pyplot as plt


# William imports to enable more stop logic
import glob
# import calculateAccuracy
import makeLinearGraphs
import sendEmail


def compute_accuracy(model, test_data, test_labels, epochs):
    batch_size = int(512 * .8)
    accuracyList = []
    correct = []
    with torch.no_grad():
        for g in range(epochs):
            randperm = torch.randperm(len(test_data))
            test_data_rand, test_labels_rand = test_data[randperm], test_labels[randperm]
            model.eval()
            for i in range(0, len(test_data), batch_size):
                mini_batch = test_data_rand[i:i + batch_size].to(device)
                mini_batch_labels = test_labels_rand[i:i + batch_size].to(device)
                pred = model(mini_batch)
                predictions = torch.round(pred)
                correct = torch.sum(predictions == mini_batch_labels)


                accuracy = correct.item()/predictions.size(0)
                accuracyList.append(accuracy)
    accuracyList = np.array(accuracyList)
    answer = accuracyList.mean()

    print(f"accuracy = {answer*100:.2f}%")
    return answer

def make_test_val(review, score):
    temp_review = []
    temp_score = []
    amount = 0
    stop = int(len(review) * 0.1)
    #for negative reviews
    for i in reversed(range(len(review))):
        if amount == stop//2:
            break
        if score[i] == 0.0:
            temp_review.append(review[i])
            temp_score.append(score[i])
            del score[i]
            del review[i]
            amount += 1
    #for positive reviews
    amount = 0
    for i in reversed(range(len(review))):
        if amount == stop//2:
            break
        if score[i] == 1.0:
            temp_review.append(review[i])
            temp_score.append(score[i])
            del score[i]
            del review[i]
            amount += 1
    return temp_review, temp_score, review, score

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def make_lists():
    csv.field_size_limit(1048576)
    review = []
    score = []
    with open('IMDB_MovieDataset.csv', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
                review.append(row['review'])  # Append review text
                if row['sentiment'] == "positive":
                    score.append(1.0)
                else:
                    score.append(0.0)

    test_review, test_score, review, score = make_test_val(review, score)
    val_review, val_score, review, score = make_test_val(review, score)
    test_score = torch.tensor(test_score).unsqueeze(dim=-1)
    val_score = torch.tensor(val_score).unsqueeze(dim=-1)

    print(len(test_review))

    return review, score, test_review, test_score, val_review, val_score

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def make_Lin_graph(train_loss, val_loss, accuracyList, trainingLoops):
    epochs = range(1, len(train_loss) + 1)  # Create a range for x-axis
   
    plt.figure(figsize=(8, 5), dpi=100)  # Set figure size
   
    # Plot training and validation loss
    plt.plot(epochs, train_loss, 'b--', label='Training Loss')  # Dashed blue line
    plt.plot(epochs, val_loss, 'r-', label='Validation Loss')   # Solid red line
    plt.plot(epochs, accuracyList, "p--", label = 'Accuracy List') # dashed purple line
   
    # Graph title and labels
    plt.title('Training loss vs. Validation Loss vs. Accuracy', fontsize=16)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
   
    # Set x-axis ticks
    plt.xticks(epochs)  
   
    # Add legend and grid
    plt.legend()
    plt.grid(True)
   
    # # Show the plot
    # plt.show()
    
    counter = 1
    while glob.glob(f"model{counter}*.png"):
        counter+=1
    fileName = f"model{counter}_{trainingLoops}loopsPlot.png"
    plt.savefig(fileName)
    print(f"Plot saved as {fileName}")
    return fileName

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def lex_order(reviews, score):
    words = set()
    for i in reviews: # problem: crashed because I traid to make another list of lists of it so I can maintain the order but that was too much for memory 1/3
        for n in i.split(): # solution found: just use the input reviews as the order and don't make a list of lists 1/7
            words.add(n)
    words = sorted(list(words))
    vocab = len(words) + 1
    hashing = {}
    order = 1
    for i in words:
        hashing[i] = order
        order += 1
    lex = []
   
   # make padding for training
    for review in reviews:
        lister = []
        for words in review.split():
            lister.append(hashing[words])
        lex.append(torch.tensor(lister))
    padded = torch.nn.utils.rnn.pad_sequence(lex, batch_first=True) # crashed: used too much ram again 6.5 million reviews was far past my computers
    score = torch.tensor(score).unsqueeze(dim=-1)
    print(f"score dim{score.size()}")
    return vocab, padded, hashing, score # solution found: use secound smaller dataset after discussing with proffessor - 1/7
#return hashing bc I want to test new data
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# class reviewer(nn.Module): 
#   def __init__(self, vocab, dim):
#     super().__init__()
#     self.emdding = nn.Embedding(vocab, dim) 
#     self.lin = nn.Linear(dim, 1)
#     self.sig = nn.Sigmoid()
#     self.drop = nn.Dropout(0.2)
#   def forward(self, x):
#     embed = self.emdding(x)
#     mean = torch.mean(embed, axis = 1)
#     lin = self.lin(mean)
#     drop = self.drop(lin)
#     sig = self.sig(drop)
#     return sig

class reviewer(nn.Module): 
    def __init__(self, vocab, dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab, dim) 
        self.lstm = nn.LSTM(dim, dim, batch_first=True)
        self.lin1 = nn.Linear(dim, 128)
        self.lin2 = nn.Linear(128, 1)
        self.drop = nn.Dropout(0.3)  # Slightly increased dropout
        self.sig = nn.Sigmoid()

    def forward(self, x):
        embed = self.embedding(x)  # Embed input
        lstm_out, _ = self.lstm(embed)  # Apply LSTM
        mean = torch.mean(lstm_out, axis=1)  # Pooling
        lin1 = self.lin1(mean)  # First Linear layer
        drop = self.drop(lin1)  # Apply Dropout
        lin2 = self.lin2(drop)  # Second Linear layer
        return self.sig(lin2)  # Sigmoid activation

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# roses are red violets are blue fix this later it crashed my laptop and this thing is new
def training_loop(vocab, padding, updated_score, val_review, val_score, test_review, test_score):
    #training loop
    model = reviewer(vocab, 256)
    model = model.to(device)
    batch_size = 128
    loss_function = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters())
    loops = 100
    losses = []
    val_losses = []
    val_batch = int(batch_size * 0.8)
    accuracyList = []
    bestAccuracy = 0.0

    for epoch in range(loops):
        if epoch % 5 == 0:
            time.sleep(60)
        startTime = time.time()
        randperm = torch.randperm(len(padding))
        padding_rand, updated_score_rand = padding[randperm], updated_score[randperm]
        model.train()
        total_loss = 0
       
        for i in range(0, len(padding), batch_size):
            mini_batch = padding_rand[i:i + batch_size]
            mini_batch = mini_batch.to(device)
            mini_batch_labels = updated_score_rand[i:i + batch_size]
            mini_batch_labels = mini_batch_labels.to(device)

            pred = model(mini_batch)
            loss = loss_function(pred, mini_batch_labels)

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
            total_loss += loss.item()


        losses.append(total_loss / (len(padding) // batch_size))
            
     #validation loop time complexity be dammed
           
        with torch.no_grad():
            val_randomperm = torch.randperm(len(val_review))
            val_review_rand = val_review[val_randomperm]
            val_score_rand = val_score[val_randomperm]
            model.eval()
            val_loss = 0
           
            for i in range(0, len(val_review), val_batch):
                mini_batch_val = val_review_rand[i:i +  val_batch].to(device)
                mini_batch_labels_val = val_score_rand[i:i + val_batch].to(device)
           
                pred = model(mini_batch_val)
                loss = loss_function(pred, mini_batch_labels_val)
                val_loss += loss.item()

            val_losses.append(val_loss / (len(val_review) // val_batch)) 
        print("loop runtime:", round(time.time() - startTime, 2))   

        accuracy = compute_accuracy(model, test_review, test_score, loops)
        print(f"Epoch {epoch}: Testing Accuracy: {accuracy:.5f}%")
        accuracyList.append(accuracy)
        newBestAccuracy = False

        if accuracy > bestAccuracy:
            bestAccuracy = accuracy
            newBestAccuracy = True
            print("New best accuracy")
        sendEmail.sendEmail(accuracy, fileName, newBestAccuracy, losses[-1], graphName, False) # trainingCompleted boolean
        print(f"Epoch {epoch}: Testing Accuracy: {accuracy:.5f}%")

    graphName = make_Lin_graph(losses, val_losses, accuracyList, loops)
    # test dataset/accuracy

    


    # if lowest is below 0.4 change how it works for later
    # new error: crashed bc I don't have 822 gb of ram in a laptop - found 1/7
    #solution found randomly picking 128 samples from the dataset and training on it - fixed 1/9
    #why does this work bc? I don't have 822 gb of ram and I know google won't let me use theirs
    saved_model = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
    checkpoint(saved_model)
    return model
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def lex_order_new(hashing, dataset):
    lex = []
    for review in dataset:
        lister = []
        for word in review:
            if word in hashing:
                lister.append(hashing[word])
        lister = torch.tensor(lister)
        lex.append(lister)  
    lex = torch.nn.utils.rnn.pad_sequence(lex, batch_first=True)
    return lex
   
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def testing_input(review, model):

    keep_going = True
    while keep_going:
        examples = input("Give me a review: ")
        examples = examples.split()
        hashing2 = lex_order_new(review, examples)
        # Let's encode these strings as numbers using the dictionary from earlier
        padder = []
        lister = []
        for word in examples:
            lister.append(hashing2[word])
        padder.append(torch.tensor(lister))

        testing_tensor = torch.nn.utils.rnn.pad_sequence(padder, batch_first=True)
        model.eval()

        if model(testing_tensor).item() < 0.5:
            print("Sounds like a negative reviewer is afoot")
        elif model(testing_tensor).item() >= 0.5:
            print("Somebody is brimming with positivity")
        else:
            print("Not to hot not to cold")
        print(model(testing_tensor))

        awnser = input("Do you want to try again?:  ")
        if awnser not in {"yes", "y"}:
            print("ARE YOU DOUBLE SURE YOU'RE DONE?")
            awnser = input()

            if awnser not in {"yes", "y"}:
                print("ok you are")
                keep_going = False


def checkpoint(state, filename="my_checkpoint2.pth.tar"):
    print("dipstick model saved")
    torch.save(state, filename)

def testing_doc(model):
    answers = []
   
    with open('IMDB Dataset.csv', encoding='utf-8') as csvfile: # reading in our testing training data
        reader = csv.DictReader(csvfile)
        lister = []
        padding = []
        for row in reader:
                review = row['review']
                if row['sentiment'] == "positive":
                    answers.append(1)
                else:
                    answers.append(0)

                for word in review:
                    lister.append(word)
                padding.append(torch.tensor(lister))
                testing_tensor = torch.nn.utils.rnn.pad_sequence(padding, batch_first=True)
                model.eval()

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

startTime = time.time()

review, score, test_review, test_score, val_review, val_score = make_lists()
print(f"takes {round(time.time() - startTime, 2)} seconds to make lists")

lexStartTime = time.time()
vocab, padded, hashing, score = lex_order(review, score)
test_review = lex_order_new(hashing, test_review)# tokenize the test review
val_review = lex_order_new(hashing, val_review)
print(f"takes {round(time.time() - lexStartTime, 2)} seconds to make the lex order")


my_model = training_loop(vocab, padded, score, val_review, val_score, test_review, test_score)


hours, remainder = divmod(int(time.time() - startTime), 3600)
minutes, seconds = divmod(remainder, 60)
print(f"total runtime: {hours}:{minutes}:{seconds}") # Output: 02:46:40



#change made: model can save itself