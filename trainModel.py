import torch

def trainModel(model, padded, updated_score, loss_function, optimizer, losses, device):
    # startTime = time.time()
    randperm = torch.randperm(len(padded))
    padded, updated_score = padded[randperm], updated_score[randperm]
    model.train() # switches to training mode, activating layers like dropout and batch normalization
    # print(torch.cuda.memory_summary(device=0, abbreviated=False))

    total_loss = 0
    for i in range(0, len(padded), 1024): # change all these back to 1024, just testing smaller chunk to decrease memory load
        mini_batch = padded[i:i + 1024]
        mini_batch = mini_batch.to(device)
        mini_batch_labels = updated_score[i:i + 1024]
        mini_batch_labels = mini_batch_labels.to(device)
        
        prediction = model(mini_batch)
        prediction = prediction.to(device)
        loss = loss_function(prediction, mini_batch_labels)

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    losses.append(total_loss / (len(padded) // 1024))
    # print("trainModel runtime:", round(time.time() - startTime,2))
    return model, optimizer, losses