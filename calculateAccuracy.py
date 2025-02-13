import torch
def calculateAccuracy(model, test_padded, test_scores, device):
    # startTime = time.time()
    model.eval()  # Switch to evaluation mode
    with torch.no_grad():  # Disable gradient computation (saves memory)
        # # Shuffle the data (optional)
        # randperm = torch.randperm(len(test_padded))
        # test_padded, test_scores = test_padded[randperm], test_scores[randperm]

        # Move tensors to the same device as the model
        test_padded = test_padded.to(device)
        test_scores = test_scores.to(device)
        
        # Process in batches to avoid out-of-memory issues
        batch_size = 1024  # Set a batch size
        total_correct = 0
        total_samples = 0
        
        for i in range(0, len(test_padded), batch_size):
            # Extract the mini-batch
            mini_batch = test_padded[i:i+batch_size]
            mini_batch_labels = test_scores[i:i+batch_size]
            
            # Make predictions
            predictions = model(mini_batch)
            
            # Look at why the model is so innacurate, it has correct classifications though -- WC 2/4
            predicted_classes = (predictions >= 0.5).float()
            print(predicted_classes[:10])
            print(mini_batch_labels[:10])
            
            # Calculate correct predictions
            total_correct += (predicted_classes == mini_batch_labels).sum().item()
            print(total_correct)
            break
            total_samples += mini_batch_labels.size(0)
        
        # Calculate accuracy
        accuracy = (total_correct / total_samples) * 100
    # print("calculate_accuracy runtime:", round(time.time() - startTime, 2))
    return accuracy