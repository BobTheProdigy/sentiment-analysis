import torch # pip install torch torchvision torchaudio
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # defaults to GPU when available, and CPU if not
import torch.nn as nn
import newLexigraphicOrder

def testing(review, model):

    keep_going = True
    while keep_going:
        examples = input("Give me a review: ")
        examples = examples.split()
        hashing2 = newLexigraphicOrder.lex_order_new(review, examples)
        # Let's encode these strings as numbers using the dictionary from earlier
        padder = []
        lister = []
        for word in examples:
            lister.append(hashing2[word])
        padder.append(torch.tensor(lister).to(device))

        testing_tensor = torch.nn.utils.rnn.pad_sequence(padder, batch_first=True).to(device)
        model.eval()

        if model(testing_tensor).item() < 0.5:
            print("Sounds like a negative reviewer is afoot")
        elif model(testing_tensor).item() >= 0.5:
            print("Somebody is brimming with positivity")
        else:
            print("oh shit")
        print(model(testing_tensor))

        # avoids breaking out of loop by accident -- WC
        answer = input("Do you want to try again?:  ")
        while answer not in {"yes", "y", "no", "n"}:
            answer = input("That is not a valid input. Do you want to try again?:  ")
        if answer not in {"yes", "y"}:
            keep_going = False