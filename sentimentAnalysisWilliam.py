import torch # pip install torch torchvision torchaudio
import torch.nn as nn
import time
import os
import sys # for printing everything to a file to save instead of to the console
# from sklearn.model_selection import train_test_split # pip install -U scikit-learn

# file imports
import calculateAccuracy
import checkpoint
import sendEmail
import makeLists
import makeLinearGraphs
import lexigraphicOrder
import trainModel
import manualTestingReviews


# Most deep learning models, including those for Natural Language Processing (NLP), require input sequences to be of the same length for efficient batch processing
# Padding ensures that all reviews in your test data have the same length
# PyTorch models only accept inputs and labels in the form of tensors
# torch defaults to loading memory onto CPU/RAM
# Model wants the two tensors (reviews and scores) to come from the same device
# when GPU offloads to CPU, it has a pointer to the info transferred to the CPU

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# "allocated memory" -- memory currently being used by a program or process
# "reserved memory" -- memory set aside for potential future use by a program
# "free memory" is the unused portion of memory that is available for allocation to any program
def get_device():
    if torch.cuda.is_available():
      allocated = torch.cuda.memory_allocated()
      reserved = torch.cuda.memory_reserved()
      total_memory = torch.cuda.get_device_properties(0).total_memory
      free_memory = total_memory - allocated - reserved

      # If GPU memory is running low, switch to CPU
      if free_memory < 0.7 * total_memory:  # Threshold for when to switch to CPU
        print("Switching to CPU due to low GPU memory")
        return torch.device('cpu')
      else:
        return torch.device('cuda:0')
    else:
      return torch.device('cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# creates NN model named reviewer
class reviewer(nn.Module): # defining a class inheriting from nn.Module
  def __init__(self, vocab, dim):
    super().__init__()
    self.emdding = nn.Embedding(vocab, dim)
    self.lin = nn.Linear(dim, 16)
    self.third = nn.Linear(16,1) # removed the need for a second hidden layer
    self.tanh = nn.Tanh()
    self.sig = nn.Sigmoid()
    self.drop = nn.Dropout(0.2)
  def forward(self, x):
    embed = self.emdding(x)
    mean = torch.mean(embed, axis = 1) # error from forgetting to add what dimentions get averaged out 1/6 -fixed 1/9
    first = self.lin(mean)
    sigmoid = self.sig(first)
    third = self.third(sigmoid)
    drop = self.drop(third)
    sig = self.sig(drop)
    return sig
  #note to self - lick balls

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

startTime = time.time()
# Load training, validation, and test data, along with their corresponding scores
trainReviews, valReviews, testReviews, trainScores, valScores, testScores = makeLists.makeLists()
trainScores = torch.Tensor(trainScores)
testScores = torch.Tensor(testScores)

trainScores = torch.unsqueeze(trainScores, dim = -1)
testScores = torch.unsqueeze(testScores, dim = -1)



# Process and pad the training & testing data
trainVocab, trainingPadded, hashing = lexigraphicOrder.lex_order(trainReviews)
testVocab, testingPadded, testHashing = lexigraphicOrder.lex_order(testReviews)


# Process and pad the training & testing data
# vocab is integer of special words, hashing is Dictionary
# trainVocab, trainingPadded, hashing = lexigraphicOrder.lex_order(trainReviews) # hashing and vocab are on the CPU, no choice
# testVocab, testingPadded, testHashing = lexigraphicOrder.lex_order(testReviews) # review and score have to come from the same device when feeding into model, result can be loaded onto gpu

# # defualts this to CPU
# trainingPadded = torch.Tensor(trainingPadded)
# testingPadded = torch.Tensor(testingPadded)
# # trainScores = torch.Tensor(trainScores)
# testScores = torch.Tensor(testScores)
# trainScores = torch.unsqueeze(trainScores, dim = -1)
# testScores = torch.unsqueeze(testScores, dim = -1)

print("completed making tensors")



# Initialize the model as reviewer object
model = reviewer(trainVocab, 256)
model = model.to(device)
loss_function = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters())

trainingLoops = int(input("desired number of training loops:"))
losses = []
previousTrainingLoops = 0
accuracyList = []


# everything to do with loading a model, only sometimes work -- 1/21 WC
loadModelString = input("Load model? ")
if (loadModelString in ["yes", "y"]):
    model, optimizer, losses, previousTrainingLoops, accuracyList = checkpoint.loadCheckpoint(model, optimizer)
    trainingLoops += previousTrainingLoops
    print("previous training loops:", previousTrainingLoops)
print(f"Beginning training {trainingLoops} loops now")


# for printing everything to a file to save instead of to the console
# with open("console_output-Piece_of_shit.txt", "w") as file:
  # Redirect stdout to the file
  # sys.stdout = file
bestAccuracy = 0.0
for trainingLoop in range(1 + previousTrainingLoops, previousTrainingLoops + trainingLoops + 1): # prints current training loop num, still trains {trainingLoop} more times
  innerStartTime = time.time()
  model, optimizer, losses = trainModel.trainModel(model, trainingPadded, trainScores, loss_function, optimizer, losses, device)
  minutes, seconds = divmod(int(time.time() - innerStartTime), 60)
  print(f"Loss after training loop {str(trainingLoop)}: {str(round(losses[-1],3))} -- runtime: {minutes}:{seconds}") # Output: "training loop {loop} -- loss:{loss} -- runtime: 46:40"

  # calculating accuracy every 5th epoch and saving model, allows for easy stopping
  if (trainingLoop % 1 == 0):
    accuracy = calculateAccuracy.calculateAccuracy(model, testingPadded, testScores, device)
    accuracyList.append(accuracy)
    print(f"Epoch {trainingLoop}: Testing Accuracy: {accuracy:.5f}%")
    newBestAccuracy = False
    
    graphName = makeLinearGraphs.makeLinearGraph(accuracyList, trainingLoop)
    savedModel = {'state_dict': model, 'optimizer': optimizer, 'losses': losses, "trainingLoops": trainingLoop}
    fileName = checkpoint.saveCheckpoint(savedModel, accuracyList, accuracy) # Saving the model and tokenizer
    if accuracy > bestAccuracy:
      bestAccuracy = accuracy
      newBestAccuracy = True
      print("New best ", end = "")
    print(f"model saved with accuracy: {bestAccuracy:.3f}%")
    # sendEmail.sendEmail(accuracy, fileName, newBestAccuracy, losses[-1], graphName, False) # trainingCompleted boolean
    
    if bestAccuracy >= 80.0:
      hours, remainder = divmod(int(time.time() - startTime), 3600)
      minutes, seconds = divmod(remainder, 60)
      print(f"runtime: {hours}:{minutes}:{seconds}") # Output: 02:46:40
      # sendEmail.sendEmail(bestAccuracy, fileName, newBestAccuracy, losses[-1], graphName True) # trainingCompleted boolean
      print("losses:", losses)
      print("accuracy:", accuracyList)
      # sys.stdout = sys.__stdout__
      os.system('shutdown -s') # was going to shut down system so it doesnt run for 4 days straight over break, stupid thing never reached

# manualTestingReviews.testing(review, model)