import torch
c=torch.load('checkpoints/gematria_60000.pt',map_location='cpu',weights_only=False)
print(f'Step: {c.get("step")}, best loss: {c.get("best_loss"):.4f}')
