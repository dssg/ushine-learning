# Ushine Learning

Ushine Learning is an API built to support [Ushahidi](ushahidi.com), a crowdsourced crisis reporting platform.

<img src="https://raw.github.com/wiki/dssg/ushine-learning/pic/logo_ushahidi.png" width="400px"/>

Before, as reports arrived to the Ushahidi platform, each report had to be reviewed and labeled by a team of humans in a highly manual process.

Now, it can first be routed to Ushine Learning for computerized detection of language, category, location, potentially sensitive information, and near-duplicate messages.

_This is a [Data Science for Social Good](http://www.dssg.io) project._

## Background

In **crisis situations** like contested elections, natural disasters, and troubling humanitarian situations, there's an _information gap_ between the _providers_ (the voters, disaster survivors, and victims) and the _responders_ (the election monitors, aid organizations, NGOs and journalists).

**Crowdsourced crisis reporting platforms**, like Ushahidi, aim to narrow this information gap. They provide centralized software to _collect, curate, and publish reports_ coming from the ground during crises.

<img src="https://raw.github.com/wiki/dssg/ushine-learning/pic/ushahidi_450_map.jpg" width="450" height="315" />

## Problem: Human review of reports doesn’t scale

Currently, each report is processed prior to publication by a human reviewer. Reviewers needs to go through a series of tasks: translating, finding the location, applying category labels, removing personally-identifying information, and more. Not only do they have to extract information, but they also need to verify its accuracy against what's truly happening on the ground.

The human review process is slow and tedious, may require domain expertise, and may be inconsistent across reviewers. It is difficult to scale and problematic for high-volume or fast-paced reporting situations.

## Solution: Automatic and suggested labels using natural-language processing

We use computing to make the review process scale. By using machine learning and natural language processing, we can make initial guesses or automatically extract items which previously had been entirely human-determined (such as categories, location, URL, and sensitive information). With our system, no longer must the reviewers do everything from scratch.

This reduces the number of reviewers needed, and lessens the time and tedium they spend processing. Instead, reviewers can focus their energies on _verifying accuracy_ and _responding_ to the reports-- the parts that really matter.

## The Project

The project has 4 major pieces. The **Machine Learning Module**, **Flask Webapp**, and **Ushahidi Plugin** make up the system's architecture. The **User Experiment** is an important part of our methodology: experimental validation of our results by testing with real users.

At the base is a Python **(1) Machine Learning Module** which learns from a corpus of labeled reports and provides automated suggested labels for novel reports. This component needs to have a way to communicate with Ushahidi, a web platform, so we've created a **(2) Flask Webapp** which which wraps the Machine Learning module and can communicate with an Ushahidi server. The Flask Webapp, at a high-level, _receives reports_ from and _sends suggestions_ to Ushahidi, using a REST-ful API and JSON objects. But the truth is that we don't talk directly to a vanilla Ushahidi; instead, we talk to an **(3) Ushahidi Plugin**. This plugin is written in PHP and deeply connected with the Ushahidi platform. It provides to glue to send and receive on the Ushahidi side.

The **(4) User Experiment** was made to test our impact on real users. Without real users, we could evaluate the accuracy of our algorithms on test data. However, the scenarios and outcomes that concerned us most were proving that we improved from "before" (no suggestions) to "after" (with machine suggestions) on parameters like: speed, accuracy, and tedium. You can read in detail about this work and our experimental results in the Wiki.

Technical details of each of these components follow:

<!-- Consider linking directly to the README.md files in subfolders -->

### **Machine Learning Module**

Location: `dssg` directory.

`machine.py` is the core class. A `machine` has methods which enable

- suggesting categories
- suggesting locations
- suggesting entities (person names, political groups, and more)
- detecting language
- detecting near-duplicate messages

The other files in this directory (`classifier.py`, `vectorizer.py`, and `platt.py`) are the classes which compose the classifier and allow category prediction. The underlying algorithm is a support vector machine, which is extensively documented in the wiki.

### **Flask Webapp**

Location: `server.py` runs the webapp and `dssg/webapp/rest_api.py` defines the API.

The webapp serves recommendations in response to POST requests, via REST. It also saves a local copy of much of the data, via SQLAlchemy.

### **Ushahidi Plugin, for deployment on an Ushahidi crowdmap instance**

Location: [dssg-integration](https://github.com/ekala/Ushahidi_Web/tree/dssg-integration)

This is a PHP plugin that for the Ushahidi platform. Note: this plugin requires some core changes into the Ushahidi platform in order to show its results. We hope these changes will be incorporated into Ushahidi 2.x and 3.0.

It functions by running the **Flask Webapp** on a server and making REST calls to the `rest_api`. These access the toolkit and machine learning functionality we provide. There is also a SQLAlchemy database that mirrors the important information from the Ushahidi app which we need for (1) updating the classifer and (2) detecting duplicate messages.

### **User Experiment**

Location: [ushine-learning-experiment](https://github.com/nathanleiby/ushine-learning-experiment)

This is a JavaScript application, built using NodeJS. It was used to evaluate the impact of machine suggestions on human performance (speed, accuracy, frustration). Note that "machine-suggested" answers in the the JSON of messages was generated using this repo's `machine` class.

## Installation Guide

### Basics

Clone the repo.

```
git clone https://github.com/dssg/ushine-learning
cd ushine-learning/
```

Install python requirements.

```
pip install -r requirements.txt
```

Install NTLK dependencies.

```
TODO
```

<!-- TODO: Can we bundle this up more simply? e.g. just install from PIP onto a fresh serer, it will fetch all the requirements? then have a `cmd` which will run it? That'd be rockin. -->

<!--
### Python Package installation

`pip install ushine-learning`
-->

### Webapp Deployment

*Configure Database and Classifier*

Create a config file.

```
cp dssg/config/dssg.ini.template dssg/config/dssg.ini
```

Edit the `dssg/config/dssg.ini` config file with
- database settings
- path to classifier, which is stored as a pickled Python object (`/path/to/classifer.pkl`), e.g. in the `dssg/data/classifier` directory.

*Run The Flask Web App*

Then, run the webapp. You can run it directly via

```
python server.py
```

To deploy the webapp in production, we suggest using [Gunicorn](http://gunicorn.org/) & [nginx](http://nginx.org/).


## Contributing to the project

To get involved, please check the [issue tracker](https://github.com/dssg/ushine-learning/issues).

To get in touch, email the team at dssg-ushahidi@googlegroups.com or file a Github issue.

## Documentation

The latest documentation is available on [ReadTheDocs](https://ushine-learning.readthedocs.org/en/latest/).

To update the documentation, you may do the following:

1. Auto-generate the latest API docs. Run `sphinx-apidoc -o doc/source dssg`, passing `-f` flag to overwrite existing apidocs.
2. Optional: Update the doc/source files directly.
3. Make the updated HTML files. Run `make html` from `doc/` path, where makefile resides.

## FAQ

**Why Ushine Learning?** Ushahidi. Machine Learning. Pronounced "oo-sheen".

## License

Copyright (C) 2013 [Data Science for Social Good Fellowship at the University of Chicago](http://dssg.io)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
