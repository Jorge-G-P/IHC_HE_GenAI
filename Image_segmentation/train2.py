
import torch

def train_model(model, dataloaders, criterion, optimizer, num_epochs, device):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, masks in dataloaders['train']:
            inputs, masks = inputs.to(device), masks.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(dataloaders['train'].dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
        
        validate_model(model, dataloaders['test'], criterion, device)

def validate_model(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, masks in dataloader:
            inputs, masks = inputs.to(device), masks.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, masks)
            val_loss += loss.item() * inputs.size(0)
    
    val_loss /= len(dataloader.dataset)
    print(f'Validation Loss: {val_loss:.4f}')