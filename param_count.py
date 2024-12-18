from models import WorldModel

model = WorldModel(loss_type='both')

def count_parameters_detailed(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.numel():,} parameters")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params:,}")
    return total_params

print(count_parameters_detailed(model))