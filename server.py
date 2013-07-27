#!/usr/bin/python
# -*- coding: utf-8 -*-

from dssg.webapp import app

app.secret_key = 'YOUR SECRET KEY'

if __name__ == "__main__":
    app.debug = True
    # load_machine()
    app.run()
