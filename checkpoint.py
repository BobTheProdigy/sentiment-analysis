import glob
import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # defaults to GPU when available, and CPU if not

def saveCheckpoint(model, accuracyList, accuracy = 0.0):
    print("saving checkpoint...")
    counter = 1
    fileName = ""
    while glob.glob(f"model{counter}*.pth.tar"): # uses glob to use astrix to allow for overwriting files to only check for model # and not epoch -- WC
        counter+=1
    fileName = f"model{counter}_{model['trainingLoops']}loops_{round(accuracy,4)}%.pth.tar"
    checkpoint = {
        'model': model['state_dict'].state_dict(),
        'optimizer': model['optimizer'].state_dict(),
        'losses': model['losses'],
        'numTrainingLoops': model['trainingLoops'],
        "accuracyList": accuracyList
    }
    torch.save(checkpoint, fileName)
    print("Model saved as", fileName)
    return fileName

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def loadCheckpoint(model, optimizer):
    
    while True:
        try:
            modelName = input("File name: ")
            print("Loading checkpoint...")
            checkpoint = torch.load(modelName, map_location = device) # map_location is where the storage should be remapped to
            break
        except FileNotFoundError:
            print(f"{modelName} is not a valid file, please try again")


    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    optimizer.load_state_dict(checkpoint['optimizer'])
    losses = checkpoint.get('losses', []) # saves an empty list if losses is missing
    accuracyList = checkpoint.get('accuracyList', [])
    # print("accuracy list:", accuracyList)
    # print("losses:", losses)
    numTrainingLoops = checkpoint['numTrainingLoops']
    print("loaded training loops:", numTrainingLoops)
    return model, optimizer, losses, numTrainingLoops, accuracyList