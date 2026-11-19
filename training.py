import torch
import torch.nn as nn


def train_step(model, optimizer, criterion, x, y):
    model.train()
    optimizer.zero_grad()
    y_pred = model(x)
    loss = criterion(y_pred, y)
    loss.backward()
    optimizer.step()
    return loss


if __name__ == "__main__":
    torch.manual_seed(42)

    model = nn.Sequential(
        nn.Linear(16, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    x = torch.randn(2, 16)
    y = torch.randint(0, 3, (2,))

    loss = train_step(model, optimizer, criterion, x, y)
    print(loss.item())


'''
Follow-up questions
The interviewer may ask you to explain:
- Logits vs. probabilities: Why does nn.CrossEntropyLoss() accept raw logits rather than softmax probabilities?
- Gradient accumulation: What happens if you forget optimizer.zero_grad()?
- Training vs. evaluation: What changes when calling model.train() versus model.eval()?
- Autograd: What does loss.backward() actually compute?
- Debugging: How would you investigate a loss that becomes NaN or fails to decrease?
'''