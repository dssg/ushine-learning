# Ushine learning: Smarter crowdsourcing for crisis maps
<img src="https://raw.github.com/wiki/dssg/ushine-learning/pic/logo_ushahidi.png" width="400px" align="right"/>

**Ushine Learning** is a machine learning API built to support [Ushahidi](http://www.ushahidi.com), a crowdsourced crisis reporting platform.

This is project is a part of the 2013 [Data Science for Social Good](http://www.dssg.io) fellowship, in partnership with [Ushahidi](http://www.ushahidi.com/).

*For a quick and gentle overview of the project, check out our [blog post](http://dssg.io/2013/07/15/ushahidi-machine-learning-for-human-rights.html).*

## Background: crisis crowdsourcing

In **crisis situations** like contested elections, natural disasters, and troubling humanitarian situations, there's an _information gap_ between the _providers_ (the voters, disaster survivors, and victims) and the _responders_ (the election monitors, aid organizations, NGOs and journalists).

**Crowdsourced crisis reporting platforms**, like Ushahidi, aim to narrow this information gap. They provide centralized software to _collect, curate, and publish reports_ coming from the ground during crises.

<img src="https://raw.github.com/wiki/dssg/ushine-learning/pic/ushahidi_450_map.jpg" width="450" height="315" />

## The problem: Human review of reports doesn’t scale

Currently, each report is processed prior to publication by a human reviewer. Reviewers needs to go through a series of tasks: translating, finding the location, applying category labels, removing personally-identifying information, and more. Not only do they have to extract information, but they also need to verify its accuracy against what's truly happening on the ground.

The human review process is slow and tedious, may require domain expertise, and may be inconsistent across reviewers. It is difficult to scale and problematic for high-volume or fast-paced reporting situations.

## The solution: annotation suggestions using natural language processing

We use computing to make the review process scale. By using machine learning and natural language processing, we can make initial guesses or automatically extract items which previously had been entirely human-determined (such as categories, location, URL, and sensitive information). With our system, no longer must the reviewers do everything from scratch.

This reduces the number of reviewers needed, and lessens the time and tedium they spend processing. Instead, reviewers can focus their energies on _verifying accuracy_ and _responding_ to the reports-- the parts that really matter.

## Project layout

Recall that we are concerned with labeling reports, and the steps a report goes through are the following:

1. A "citizen" submits a report to Ushahidi.
2. **NEW**: Ushahidi sends the report to Ushine Learning, which generates suggested labels and returns them to Ushahidi.
3. Ushahidi shows the incoming report to an "admin", who annotates it. **NEW**: Suggested labels can be shown to the admin to help make their annotation process easier.
4. The admin applies the final labels and approves the report.
5. This report, with its labels, is added to a map of all reports. This map is used to help raise situational awareness.

In order to achieve this workflow, our project has 4 major pieces. The **Machine Learning Module**, **Flask Webapp**, and **Ushahidi Plugin** make up the system's architecture. The **User Experiment** is an important part of our methodology: experimental validation of our results by testing with real users.

At the base is a Python **(1) Machine Learning Module** which learns from a corpus of labeled reports and provides automated suggested labels for novel reports. This component needs to have a way to communicate with Ushahidi, a web platform, so we've created a **(2) Flask Webapp** which which wraps the Machine Learning module and can communicate with an Ushahidi server. The Flask Webapp, at a high-level, _receives reports_ from and _sends suggestions_ to Ushahidi, using a REST-ful API and JSON objects. But the truth is that we don't talk directly to a vanilla Ushahidi; instead, we talk to an **(3) Ushahidi Plugin** deployed on a Crowdmap instance. This plugin is written in PHP and connected with the Ushahidi Crowdmap. It provides to glue to send and receive on the Ushahidi side. (Note: this plugin requires some core changes into the Ushahidi platform in order to show its results. We hope these changes will be incorporated into Ushahidi 2.x and 3.0.)

The **(4) User Experiment** was made to test our impact on real users. Without real users, we could evaluate the accuracy of our algorithms on test data. However, the scenarios and outcomes that concerned us most were proving that we improved from "before" (no suggestions) to "after" (with machine suggestions) on parameters like: speed, accuracy, and frustration. You can read in detail about this work and our experimental results in the Wiki.

Technical details of each of these components are linked below.

1. [Machine Learning Module](https://github.com/dssg/ushine-learning/tree/master/dssg)
2. [Flask Webapp](https://github.com/dssg/ushine-learning/tree/master/dssg/webapp)
3. [Ushahidi Plugin](https://github.com/ekala/Ushahidi_Web/tree/dssg-integration)
4. [User Experiment](https://github.com/nathanleiby/ushine-learning-experiment)

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
mv nltk_data /usr/share/nltk_data # on unix
```

<!-- TODO: Can we bundle this up more simply? e.g. just install from PIP onto a fresh serer, it will fetch all the requirements and install NLTK dependencies? then have a `cmd` which will run it? That'd be rockin.

Should work on Ubuntu and Amazon-Linux at a minimum, since these are most likely to host Ushahidi instances. -->

<!--
### Python Package installation

`pip install ushine-learning`
-->

### Webapp Deployment

*Setup configuration*

Create a config file.

```
cp dssg/config/dssg.ini.template dssg/config/dssg.ini
```

Edit the `dssg/config/dssg.ini` config file with
- database settings
- path to classifier, which is stored as a pickled Python object (`/path/to/classifer.pkl`), e.g. in the `dssg/data/classifier` directory.

*How To Run The Flask Web App*

Then, run the webapp. You can run it directly via

```
python server.py
```

To deploy the webapp in production, we suggest using [Gunicorn](http://gunicorn.org/) & [nginx](http://nginx.org/).

## Documentation

The latest documentation is available on [ReadTheDocs](https://ushine-learning.readthedocs.org/en/latest/).

To update the documentation, you may do the following:

1. Auto-generate the latest API docs. Run `sphinx-apidoc -o doc/source dssg`, passing `-f` flag to overwrite existing apidocs.
2. Optional: Update the doc/source files directly.
3. Make the updated HTML files. Run `make html` from `doc/` path, where makefile resides.

## FAQ

**Why Ushine Learning?** Ushahidi. Machine Learning. Pronounced "oo-sheen".


## Team
![Team](https://raw.github.com/dssg/dssg.github.io/761993c24ea2991170ef64048115cb805f5f13fb/img/people/teams/ushahidi.png)


## Contributing to the project

To get involved, please check the [issue tracker](https://github.com/dssg/ushine-learning/issues). Issues include everything from bugs to new project ideas that we'd like to see happen!

To get in touch, email the team at dssg-ushahidi@googlegroups.com or file a Github issue.

## License

The MIT License (MIT)

Copyright (C) 2013 [Data Science for Social Good Fellowship at the University of Chicago](http://dssg.io)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
