import torch


def list_arrays_to_tensor(arrays):
    return [torch.tensor(arr).float() for arr in arrays]


def manifold_shift_image(img):
    _, w = img.shape[-2:]
    w_shift = torch.randint(low=0, high=w, size=(1,))
    return torch.roll(img, (0, w_shift), (-2, -1))


def flip_image(img):
    return torch.flip(img, dims=[2])


def fft2(X_t: torch.Tensor) -> torch.Tensor:
    reshaped_X_t = torch.permute(X_t, (1,0,2,3))
    reshaped_X_f = torch.vmap(torch.fft.fft2)(reshaped_X_t)
    X_f = torch.permute(reshaped_X_f, (1, 0, 2, 3))
    return X_f


def ifft2(X_f: torch.Tensor) -> torch.Tensor:
    reshaped_X_f = torch.permute(X_f, (1,0,2,3))
    reshaped_X_t = torch.vmap(torch.fft.ifft2)(reshaped_X_f)
    X_t = torch.permute(reshaped_X_t.real, (1, 0, 2, 3))
    return X_t


def compute_accuracy(logits, targets):
    predictions = torch.argmax(logits, dim=1)
    correct = (predictions == targets).sum().item()
    accuracy = correct / targets.size(0) * 100  
    return accuracy


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def split_data(dim, train_test_r=.9, train_val_r=.976, seed=None):
    rng = torch.Generator().manual_seed(seed)
    train_max = int(train_test_r*train_val_r*dim)
    val_max = int(train_test_r*dim)
    indices = torch.randperm(dim, generator=rng)
    return {
         'train': indices[:train_max], 
         'val': indices[train_max:val_max],
         'test': indices[val_max:],
    }