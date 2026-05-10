import torch

class TextEncoder:
    def __init__(self, alphabet: str):
        self.alphabet = alphabet
        self.char2idx = {c: i for i, c in enumerate(alphabet)}
        self.idx2char = {i: c for i, c in enumerate(alphabet)}

    def encode(self, text: str) -> torch.Tensor:
        return torch.tensor(
            [self.char2idx[c] for c in text],
            dtype=torch.long
        )

    def __len__(self):
        return len(self.alphabet)

def collate_fn(batch, encoder: TextEncoder):
    """
    batch: List[(image, text_string)]
    """
    # print(batch)
    _, images, texts = zip(*batch)

    B = len(images)
    C, H = images[0].shape[:2]
    widths = [img.shape[2] for img in images]
    max_w = max(widths)

    # pad images
    imgs = torch.zeros(B, C, H, max_w)
    for i, img in enumerate(images):
        imgs[i, :, :, :img.shape[2]] = img

    # encode texts
    encoded = [encoder.encode(t) for t in texts]
    lengths = torch.tensor([len(e) for e in encoded], dtype=torch.long)
    max_len = max(lengths)

    targets = torch.full(
        (B, max_len),
        fill_value=-1,
        dtype=torch.long
    )
    for i, e in enumerate(encoded):
        targets[i, :len(e)] = e

    return imgs, targets, lengths
