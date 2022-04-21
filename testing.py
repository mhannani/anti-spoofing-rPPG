import torch.nn


class Name(torch.nn.Module):
    def __init(self):
        super().__init__()

    def __format__(self, format_spec):
        pass


if __name__ == "__main__":
    name = Name()
    print(next(name.parameters()).is_cuda)
