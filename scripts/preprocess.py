from torch.utils.data import DataLoader, random_split
from torchvision import datasets

print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("Available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

device = torch.device('cuda:0') 


features = 28*28
classes = 10

transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])

train_dataset = datasets.FashionMNIST(
    root="../../src/data",
    train=True,
    download=True,
    transform=transform,
)
train_size, val_size = int(len(train_dataset) * 0.9), int(len(train_dataset) - len(train_dataset) * 0.9)
train_dataset_s, val_dataset_s = random_split(
    train_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

test_dataset = datasets.FashionMNIST(
    root="../../src/data",
    train=False,
    download=True,
    transform=transform,
)

batch_size = 128

train_loader = DataLoader(train_dataset_s, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset_s, batch_size=batch_size, shuffle=False, drop_last=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)