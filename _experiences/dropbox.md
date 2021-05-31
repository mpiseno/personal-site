---
layout: page
title: Dropbox
position_name: Software Engineer Intern
permalink: /dropbox/
description: Summer 2020
img: /assets/img/dropbox.jpeg
importance: 1
dates: May - Aug 2020
public: True
---

<img src="/assets/img/projects/anxiety.png" style="max-width: 100%;">

If the above image gives you anxiety, then do I have an intern project for you! Summer 2020 I worked at Dropbox, where I developed the ML infrastructure for a chrome extension that helps people keep their tabs organized and less cluttered. Specifically, I created a system on top of the chrome extension that periodically classifies each tab as "junk" (meaning it can be closed) or "not junk".

<center>
    <img src="/assets/img/projects/junk-not-junk.png" style="max-width: 70%; padding: 5%" alt="iykyk">
</center>

It did this by sending data collected during interaction with the tab to a server where a trained neural network performed the classification then sent the predictions back to the user. Let's take a deeper dive to see how it works.

## ML for Junk Tab Classification

First, I had to construct a training dataset for the ML model by sending real (anonymized) user data about tab usage to a server. This required that I quickly build the infrastructure for logging data and release a version of the chrome extension to our (small at the time) set of users. Of course, this did introduce bias into the data set since our user base was mostly other people at Dropbox, but we can always retrain the model on a larger dataset. Once the dataset was created, it was time to build the actual ML pipeline, which looks like this:

<center>
    <img src="/assets/img/projects/junk-prediction-system.png" style="max-width: 80%; padding: 5%" alt="iykyk">
</center>

1. First, every 10 minutes or so, data about each tab is sent to a server. This data includes things like the hashed url, active time spent on each tab, time since the tab was last used, etc.
2. Next, the raw data that was sent to the server in step 1 is preprocessed into feature vectors for the neural network. A separate server is running a process that waits on requests, which contain the feature vectors as a payload, from the first server and gives them to the neural network.
3. Onces the neural network outputs its predictions about each tab, they get routed back to the client (i.e. the user's browser), where a nice UI tells the user whether or not they have junk tabs.


## Conclusion

This project involved a ton of moving parts, and I had to write thousands of lines of code to build the all infrastruture. But other than learning Typescript and how to do ML in a production setting, this project taught me how to manage a project timeline effectively. I had to build the infrstructure to log real user data quickly so that I had time to collect enough data to train the neural network. While the data was being collcted from real users, I had to build the rest of the ML infrastructure, and still make time for HackWeek (a week-long Dropbox internal Hackathon).

Overall interning at Dropbox was an amazing and rewarding experience! The project's success would not have been possible without my awesome team, especially my phenomenal mentor Tom.


