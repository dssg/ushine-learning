Flask Webapp
====

`server.py` runs the webapp and `dssg/webapp/rest_api.py` defines the API.

The webapp serves recommendations in response to POST requests, via REST. It also saves a local copy of much of the data, via SQLAlchemy.

 These access the toolkit and machine learning functionality we provide. There is also a SQLAlchemy database that mirrors the important information from the Ushahidi app which we need for (1) updating the classifer and (2) detecting duplicate messages.
