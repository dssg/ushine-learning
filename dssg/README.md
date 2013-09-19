Machine Learning Module
====

`machine.py` is the core class. A `machine` has methods which enable

- suggesting categories
- suggesting locations
- suggesting entities (person names, political groups, and more)
- detecting language
- detecting near-duplicate messages

The other files in this directory (`classifier.py`, `vectorizer.py`, and `platt.py`) are the classes which compose the classifier and allow category prediction. The underlying algorithm is a support vector machine, which is extensively documented in the wiki.