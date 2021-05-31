---
layout: page
title: Facebook AI Research
position_name: Software Engineer Intern
description: Winter 2020
permalink: /facebook/
img: /assets/img/fb.jpeg
importance: 2
dates: Jan - March 2020
public: True
---

## Habitat: A Platform for Embodied AI Research

[Habitat](https://aihabitat.org/) is a platform for embodied AI research. Embodied AI is the study of intelligent systems that have a physical or virtual embodiment. Robots are a common example of embodied AI, as they physically move around and make complex decisions within their environments.

<center>
    <img src="/assets/img/projects/facebook/habitat.gif" style="width: 80%; padding: 5%">
</center>

Habitat was created by Facebook AI Research (FAIR) to excellerate research in embodied AI. It comes with several indoor environments, popular algorithms implemented as baselines such as PPO and TRPO, and an annual challenge, the "Habitat Challenge", which was created to benchmark and accelerate progress in embodied AI. I interned on the Habitat team at FAIR in Winter 2020 where I developed a research tool for extracting images from within Habitat environments.

## Image Extractor

Generating and labelling real world image data is time consuming and expensive. Using Habitat to automatically generate images from within Habitat's virtual environments provides researchers with a much cheaper and easier option, as the images already come with semantic labels.

Using the image extractor is very simple - you only need to provide the constructor with a filepath to the meshfile used for the environment. Here is a short example of instantiating an image extractor and displaying a few RGB, depth, and semantic images:

```python
import numpy as np
import matplotlib.pyplot as plt

from habitat_sim.utils.data import ImageExtractor


# For viewing the extractor output
def display_sample(sample):
    img = sample["rgba"]
    depth = sample["depth"]
    semantic = sample["semantic"]

    arr = [img, depth, semantic]
    titles = ["rgba", "depth", "semantic"]
    plt.figure(figsize=(12, 8))
    for i, data in enumerate(arr):
        ax = plt.subplot(1, 3, i + 1)
        ax.axis("off")
        ax.set_title(titles[i])
        plt.imshow(data)

    plt.show()


scene_filepath = "data/scene_datasets/habitat-test-scenes/apartment_1.glb"

extractor = ImageExtractor(
    scene_filepath,
    img_size=(512, 512),
    output=["rgba", "depth", "semantic"],
)

# Index in to the extractor like a normal python list
sample = extractor[0]

# Or use slicing
samples = extractor[1:4]
for sample in samples:
    display_sample(sample)

# Close the extractor so we can instantiate another one later
# (see close method for detailed explanation)
extractor.close()
```

Output:

<center>
    <img src="/assets/img/projects/facebook/extractor-example-output.png" style="max-width: 80%;">
</center>



## Use Case: Semantic Segmentation

As a proof of concept for using the image extractor to solve a task involving image data, I created an end-to-end data pipeline for learning a semantic segmentation task on indoor image data using UNet, a popular architecture for semantic segmentation. Here is an example of incorporating the image extractor into [PyTorch Datasets and Dataloaders](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html).

```python
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import utils
from torchvision.transforms import ToTensor

from habitat_sim.utils.data import ImageExtractor


# Replace with the path to your scene file
SCENE_FILEPATH = 'data/scene_datasets/habitat-test-scenes/apartment_0/mesh.ply'
BATCH_SIZE = 4

class SemanticSegmentationDataset(Dataset):
    def __init__(self, extractor, transforms=None):
        # Define an ImageExtractor
        self.extractor = extractor

        # We will perform preprocessing transforms on the data
        self.transforms = transforms

        # Habitat sim outputs instance id's from the semantic sensor (i.e. two
        # different chairs will be marked with different id's). So we need
        # to create a mapping from these instance id to the class labels we
        # want to predict. We will use the below dictionaries to define a
        # funtion that takes the raw output of the semantic sensor and creates
        # a 2d numpy array of out class labels.
        self.labels = {
            'background': 0,
            'wall': 1,
            'floor': 2,
            'ceiling': 3,
            'chair': 4,
            'table': 5,
        }
        self.instance_id_to_name = self.extractor.instance_id_to_name
        self.map_to_class_labels = np.vectorize(
            lambda x: self.labels.get(self.instance_id_to_name.get(x, 0), 0)
        )

    def __len__(self):
        return len(self.extractor)

    def __getitem__(self, idx):
        sample = self.extractor[idx]
        raw_semantic_output = sample['semantic']
        truth_mask = self.get_class_labels(raw_semantic_output)

        output = {
            'rgb': sample['rgba'][:, :, :3],
            'truth': truth_mask.astype(int),
        }

        if self.transforms:
            output['rgb'] = self.transforms(output['rgb'])
            output['truth'] = self.transforms(output['truth']).squeeze(0)

        return output

    def get_class_labels(self, raw_semantic_output):
        return self.map_to_class_labels(raw_semantic_output)


extractor = ImageExtractor(SCENE_FILEPATH, output=['rgba', 'semantic'])

dataset = SemanticSegmentationDataset(extractor,
    transforms=transforms.Compose([transforms.ToTensor()])
)

# Create a Dataloader to batch and shuffle our data
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
```

The full code for this example can be seen in the image extractor [documentation](https://aihabitat.org/docs/habitat-sim/image-extractor.html). Here are the results for overfitting on a small subset of images extracted from the [Replica Dataset](https://github.com/facebookresearch/Replica-Dataset):

<center>
    <img src="/assets/img/projects/facebook/semantic-segmentation-results.png" style="max-width: 80%;">
</center>

The top row contains RGB images from within Habitat. The middle row is the ground-truth semantic mask. The bottom row is the model's predicted semantic mask.


## Conclusion

My internship at FAIR was a fantastic learning experience, as I collaborated with researchers and engineers to develop a useful research tool and documented it extensively. Make sure to check out the full [docs](https://aihabitat.org/docs/habitat-sim/image-extractor.html) for an in-depth look at the image extractor API and semantic segmentation example!




