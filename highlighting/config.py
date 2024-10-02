import torch

device = torch.device("cuda:1")

templates =  [f"{t}\n" for t in [
    "This is what I mean when I talk about {}.",
    "{} is a pretty good title for this.",
    "I found out that {} is about the following.",
    "The things most commonly associated with {} are these.",
    "I think this is about {}.",
    "Whatever you say, this is about {}.",
    "And here's what reminds me of {}.",
    "{} screams from every corner of this.",
    "If {} were a flavor, this would taste like it.",
    "This dances to the rhythm of {}.",
    "This has {} written all over it.",
    "You can smell {} all over this.",
    "If {} were a book, this would be its intro.",
    "This is what {} looks like in reality.",
    "This makes {} look like a close relative topic.",
    ]
]

