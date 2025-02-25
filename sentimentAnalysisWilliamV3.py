import csv
import torch
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import time
import os

# File imports
import trainModel
import lexigraphicOrder
import checkpoints
import makeLists
import makeLinearGraphs
import calculateAccuracy
import sendEmail
import manualTestingReviews

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class reviewer(nn.Module): # new porblem found: high af loss 1/10
  def __init__(self, vocab, dim):
    super().__init__()
    self.emdding = nn.Embedding(vocab, dim) # caused error with space used 1/6 - fixed 1/7 (error is from training not module)
    self.lin = nn.Linear(dim, 1)
    self.lin2 = nn.Linear(16, 1)
    self.sig = nn.Sigmoid()
  def forward(self, x):
    embed = self.emdding(x)
    mean = torch.mean(embed, axis = 1)
    sigs = self.sig(mean)
    lin = self.lin(sigs)
    sig = self.sig(lin)
    return sig
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def testing_input(review, model):
    keep_going = True
    while keep_going:
        examples = input("Give me a review: ")
        examples = examples.split()
        hashing2 = lexigraphicOrder.lex_order_new(review, examples)
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
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
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

# make lists code
start_lister = time.time()
review, score, test_review, test_score, val_review, val_score = makeLists.make_lists()
end_lister = time.time()
print(f"takes {round(end_lister - start_lister, 2)} seconds to make lists")

# lexigraphic order code -- turns words into numbers for NN
start_lex = time.time()
vocab, padded, hashing, score = lexigraphicOrder.lex_order(review, score)
padd_width = len(padded[0])
test_review = lexigraphicOrder.lex_order_new(hashing, test_review, padd_width)# tokenize the test review
val_review = lexigraphicOrder.lex_order_new(hashing, val_review, padd_width)
end_lex = time.time()
print(f"takes {round(end_lex - start_lex, 2)} seconds to make the lex order")

# moved some of training loop function out to allow loadCheckpoint to work
model = reviewer(vocab, 256)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters())
trainingLosses = []
validationLosses = []
previousTrainingLoops = 0
manuallyTestReviews = True

# loads model and optimizer from checkpoint
loadCheckpoint = input("Would you like to load a checkpoint? (y/n): ")
if (loadCheckpoint == 'y'):
    [model, optimizer, trainingLosses, validationLosses, previousTrainingLoops] = checkpoints.loadCheckpoint(model, optimizer, device)
start_model = time.time()
checkpoint = trainModel.training_loop(model, optimizer, vocab, padded, score, val_review, val_score, test_review, test_score, trainingLosses, validationLosses, previousTrainingLoops, device)
end_model = time.time()
print(f"takes {end_model - start_model} seconds for training")

loops = checkpoint['loops']
trainingLosses = checkpoint['trainingLosses']
validationLosses = checkpoint['validationLosses']
checkpoint['model'] = model.state_dict() # doesn't change in training_loop so just adding to checkpoint manually
checkpoint['optimizer'] = optimizer.state_dict() # doesn't change in training_loop so just adding to checkpoint manually
if loops != previousTrainingLoops: # only runs if training loops > 0
    print('running line 127')
    pngFileName = makeLinearGraphs.makeLinearGraph(trainingLosses, validationLosses, loops) # make plot of training and validation from loops
    accuracy = calculateAccuracy.compute_accuracy(model, test_review, test_score, loops, device) # test dataset/accuracy
    try:
        sendEmail.sendEmail(loops, round(trainingLosses[-1], 2), round(validationLosses[-1], 2), True, pngFileName, accuracy)
    except:
        print("unable to connect to server")
    checkpoints.saveCheckpoint(checkpoint, loops)
if manuallyTestReviews:
    manualTestingReviews.testing_input(review, model, padd_width) # Broken, need to fix

os.system('shutdown -s')
print("done")
