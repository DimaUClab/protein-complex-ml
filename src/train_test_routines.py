import torch
from criterions import MaskedMCC

def train_one_epoch(model, dataloader, criterion, optimizer, device, verbose=False):
    '''Trains a model one epoch across an entire data loader. Returns the average loss.'''
    model.train()
    total_loss = 0
    
    for i, batch in enumerate(dataloader):
        # Move all batch data to the active device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        optimizer.zero_grad()
        outputs = model(
            input1_ids=batch['input1_ids'],
            input1_attention_mask=batch['input1_attention_mask'],
            input2_ids=batch['input2_ids'],
            input2_attention_mask=batch['input2_attention_mask']
        )
        
        # Criterion already handles masking internally
        loss = criterion(outputs, batch['labels'], batch['mask'])
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if verbose and i % 10 == 0:
            print(f"Batch {i} Loss: {loss.item():.4f}")
            
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    '''
    Evaluates a model. Returns average loss and MCC.
    '''
    model.eval()
    total_loss = 0

    # Initialize and move metric to device
    val_mcc = MaskedMCC().to(device)
    val_mcc.reset() # Good practice to ensure clean state

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(
                input1_ids=batch['input1_ids'],
                input1_attention_mask=batch['input1_attention_mask'],
                input2_ids=batch['input2_ids'],
                input2_attention_mask=batch['input2_attention_mask']
            )
            
            # Loss calculation
            loss = criterion(outputs, batch['labels'], batch['mask'])
            total_loss += loss.item()
            
            # Metric update: Use sigmoid because BCEWithLogitsLoss inputs raw logits
            val_mcc.update(torch.sigmoid(outputs), batch['labels'], batch['mask'])
    
    # Compute final metric and move back to CPU for logging/Optuna
    final_mcc = val_mcc.compute().item()
    avg_loss = total_loss / len(dataloader)
    
    print(f"Validation - Loss: {avg_loss:.4f}, MCC: {final_mcc:.4f}")
    return avg_loss, final_mcc