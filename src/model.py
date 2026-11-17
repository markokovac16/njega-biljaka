"""Model, transformacije slika i učitavanje podataka. Koriste ga treniranje, evaluacija i API."""
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision.transforms import v2 as T

import config as c

SREDINA, DEVIJACIJA = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  # ImageNet


def uredaj():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def napravi_model(predtreniran=True):
    """EfficientNet-B0 predtreniran na ImageNetu kojemu je zadnji sloj zamijenjen slojem s 2 izlaza."""
    m = models.efficientnet_b0(weights="IMAGENET1K_V1" if predtreniran else None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, len(c.RAZREDI))
    return m


def _zavrsetak():
    return [T.ToImage(), T.ToDtype(torch.float32, scale=True), T.Normalize(SREDINA, DEVIJACIJA)]


SIVA = tuple(round(255 * x) for x in SREDINA)   # prosječna boja ImageNeta: nakon normalizacije rub je nula


def s_rubom(slika):
    return ImageOps.pad(slika, (c.VELICINA, c.VELICINA), color=SIVA)


def transformacija_evaluacija(nacin="sredisnji_izrez"):
    """Svođenje slike na kvadrat 224 x 224 bez nasumičnih izmjena.

    sredisnji_izrez  kraća stranica na 256 piksela, pa kvadrat iz sredine (rubovi izduženih slika se odbace)
    razvucena        cijela slika razvučena u kvadrat (mijenja omjer stranica)
    s_rubom          cijela slika smanjena i dopunjena sivim rubom do kvadrata
    """
    if nacin == "sredisnji_izrez":
        koraci = [T.Resize(256), T.CenterCrop(c.VELICINA)]
    elif nacin == "razvucena":
        koraci = [T.Resize((c.VELICINA, c.VELICINA))]
    elif nacin == "s_rubom":
        koraci = [T.Lambda(s_rubom)]
    else:
        raise ValueError(nacin)
    return T.Compose([*koraci, *_zavrsetak()])


def transformacija_trening(vrsta="osnovna"):
    if vrsta == "osnovna":
        koraci = [
            T.RandomResizedCrop(c.VELICINA, scale=(0.6, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.2, 0.2, 0.2, 0.02),
        ]
    elif vrsta == "jaka":
        # jača augmentacija: manji isječci, zrcaljenja, RandAugment, zamućenje, sivi tonovi
        koraci = [
            T.RandomResizedCrop(c.VELICINA, scale=(0.25, 1.0), ratio=(0.6, 1.6)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandAugment(num_ops=2, magnitude=9),
            T.RandomApply([T.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.2),
            T.RandomGrayscale(p=0.05),
        ]
    else:
        raise ValueError(vrsta)
    return T.Compose([*koraci, *_zavrsetak()])


class Slike(Dataset):
    """Čita smanjene kopije slika prema recima manifesta."""

    def __init__(self, df: pd.DataFrame, transformacija):
        self.kopije = df.kopija.tolist()
        self.oznake = (df.oznaka == c.N).astype(int).tolist()
        self.transformacija = transformacija

    def __len__(self):
        return len(self.kopije)

    def __getitem__(self, i):
        with Image.open(c.PREDMEMORIJA / self.kopije[i]) as im:
            x = self.transformacija(im.convert("RGB"))
        return x, self.oznake[i]


def ucitavac(df, transformacija, mijesaj=False, serija=64):
    return DataLoader(Slike(df, transformacija), batch_size=serija, shuffle=mijesaj,
                      num_workers=6, persistent_workers=False, drop_last=mijesaj)


@torch.no_grad()
def logiti(model, df, dev=None, znacajke=False, transformacija=None):
    """Vraća logite (ili značajke predzadnjeg sloja) za sve slike iz df, redom."""
    dev = dev or uredaj()
    model.eval().to(dev)
    izlazi = []
    for x, _ in ucitavac(df, transformacija or transformacija_evaluacija(), serija=128):
        x = x.to(dev)
        if znacajke:
            y = torch.flatten(model.avgpool(model.features(x)), 1)
        else:
            y = model(x)
        izlazi.append(y.float().cpu().numpy())
    return np.concatenate(izlazi)


def ucitaj_model(putanja, dev=None):
    stanje = torch.load(putanja, map_location="cpu", weights_only=True)
    m = napravi_model(predtreniran=False)
    m.load_state_dict(stanje["tezine"])
    return m.eval().to(dev or uredaj()), stanje
