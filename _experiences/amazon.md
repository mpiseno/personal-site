---
layout: page
title: Amazon
position_name: Software Engineer Intern
permalink: /amazon/
description: Summer 2019
img: /assets/img/amazon.png
importance: 3
dates: May - Aug 2019
public: True
---

## A9 Visual Search

In summer 2019, I worked at A9, a former subsidiary of Amazon that is now part of Amazon. I worked on the Visual Search & Augmented Reality team, where I developed algorithms to detect linear boundaries indoors. This is important for the Amazon shopping app: when a user wants to view a product in their room using the AR feature in the app, it is necessary to determine where certain planes (e.g. the floor and wall) are.

<center>
    <img src="/assets/img/projects/amazon/view-in-room.gif" style="max-width: 80%; padding: 5%">
</center>

While at A9, I developed a method for automatically detecting the boundary between these planes. As AR for shopping becomes more popular, this method will be useful to improve the customers' shopping experiences so they do not have to manually define that boundary like above.


## Edge Detection and Linear Segmentation

The method for finding the correct linear boundary between two planes of interest in a room can be broken down into two parts - line detection and line classification.

1. Line Detection: The algorithm makes use of the Hough transform along with a few preprocessing methods such as gaussian blurring.
2. Line Classification: After we have found several "candidate lines" from the Hough transform, we pass the image along with candidate lines through a CNN trained to determine whether the candidate lines are the true boundary we care about or not. This relies on a method called linear pooling, in which we pool layer activations along the linear boundaries defined by the candidate lines.

After assigning each candidate line found by the Hough transform a probability of being the correct line, the algorithm predicts the true boundary to be the candidate line with the highest probability.

<center>
    <img src="/assets/img/projects/amazon/algorithm.png" style="max-width: 90%; padding: 5%">
</center>

### Performance Metrics

Once the algorithm predicts a line for the true boundary, that line segments the image into two regions P and Q that are subsets of the image.

<center>
    <img src="/assets/img/projects/amazon/segment.png" style="max-width: 70%; padding: 5%">
</center>

To evaluate the performance of the model, we use intersection over union (IoU) with varying thresholds compared with the ground truth $$P_{gt}$$ and $$Q_{gt}$$. We can then define an accuracy measure by assigning predicted lines that produce an IoU score greater than a threshold $$\tau \in [0, 1]$$ to be correct.




