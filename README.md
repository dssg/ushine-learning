# Ushine Learning

With machine learning and natural language processing, we’re building open-source tools to improve the human review process of crisis reports.

This is a [Data Science for Social Good](http://www.dssg.io) project.

## The Problem

In **crisis situations** like contested elections, natural disasters, and troubling humanitarian situations, there's an **information gap** between the **information providers** (the voters, disaster survivors, and victims) and the **information responders** (the election monitors, aid organizations, NGOs and journalists).

**Crowdsourced crisis reporting platforms**, like [Ushahidi](ushahidi.com) and others, aim to narrow this information gap. They provide centralized software to **collect, curate, and publish reports** coming from the ground.

<!--
TODO: ![Ushahidi Logo](http://www.no-straight-lines.com/wp-content/uploads/2013/01/logo_ushahidi.png)
-->

Currently, each report is processed prior to publication by a human reviewer who ensures the quality of the report. This human process is slow and tedious. It also may require domain expertise and be inconsistent across reviewers.It is difficult to scale, and requires more volunteers to handle an increasing number and rate of incoming reports.

[Read more about Crowdsourced Crisis Reporting...](wiki/Crowdsourced Crisis Reporting)

## The Solution

We use computing to make the review process scale. By using machine learning and natural language processing, we can make initial guesses or automatically extract items which previously had been entirely human-determined (such as categories, location, URL, and sensitive information). With our system, no longer must the reviewers do everything from scratch. This reduces the number of reviewers needed, and lessens the time and tedium they spend processing. Instead, reviewers can focus their energies on _verifying accuracy_ and _responding_ to the reports-- the parts that really matter.

## The Project
There are multiple components to the project:

### **Machine Learning module**

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

### **Experiment**

Location: [ushine-learning-experiment](https://github.com/nathanleiby/ushine-learning-experiment)

This is a JavaScript application, built using NodeJS. It was used to evaluate the impact of machine suggestions on human performance (speed, accuracy, frustration). Note that "machine-suggested" answers in the the JSON of messages was generated using this repo's `machine` class.

## Installation Guide

First you will need to clone the repo.

```
git clone https://github.com/dssg/ushine-learning
cd ushine-learning/
```

<!--
### Python Package installation

`pip install ushine-learning`
-->

### Webapp Deployment

*How To Run The Flask Web App*

Create a config file.

```
cp dssg/config/dssg.ini.template dssg/config/dssg.ini
```

Edit the `dssg/config/dssg.ini` config file with
- database settings
- path to classifier, which is stored as a pickled Python object (`/path/to/classifer.pkl`), e.g. in the `dssg/data/classifier` directory.

Then, run the webapp. You can run it directly via

```
python server.py
```

To deploy the webapp in production, we suggest using [Gunicorn](http://gunicorn.org/) & [nginx](http://nginx.org/).


## Contributing to the project
To get involved, please check the [issue tracker](https://github.com/dssg/ushine-learning/issues).

To get in touch, email the team at dssg-ushahidi@googlegroups.com.

### Documentation

The latest documentation is available on [ReadTheDocs](https://ushine-learning.readthedocs.org/en/latest/).

To update the documentation, you may do the following:

_First, auto-generate the latest API docs_

```
sphinx-apidoc -o doc/source dssg # -f to overwrite

# sphinx-apidoc -o doc/source dssg -f && cd doc && make html && cd ..
```

_Optionally: update the doc/source files directly_

_Finally, make HTML (from `docs/` path, where makefile resides)_

```
make html
```

## License

Copyright (C) 2013 [Data Science for Social Good Fellowship at the University of Chicago](http://dssg.io)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
