from torch.utils.data import DataLoader
from utils.inference import inference
from utils.NPZ_loader import NPZLoader
from utils.split import get_sets

if __name__ == "__main__":
    # Get data
    dataset = NPZLoader('./Data')

    # split dataset
    _, test_set = get_sets(dataset, 0.8)

    # Get train dataloader
    train_data = DataLoader(test_set, batch_size=5)

    a, b, c, d = next(iter(train_data))
    a = a.to('cuda:0')

    # inference
    print(inference(a, 12))
