import logging as logger
from flask import abort, jsonify, request

from dssg import db
from dssg.Machine import Machine
from dssg.model import *
from dssg.webapp import app


@app.route('/v1/language', methods=['POST'])
def detect_language():
    """Given some text, returns a ranked list of likey natural languages
    the given content is in
    
    Input parameters:
        text: string
    """
    if not request.json or not 'text' in request.json:
        abort(400)
    language = Machine.guess_language(request.json['text'])[0]
    
    return jsonify({'language': language[0], "confidence": language[1]})

@app.route('/v1/deployments', methods=['POST'])
def add_deployment():
    """Registers a new Ushahidi deployment. The following information
    should be included in the request
        - deployment name
        - list of categories; <id, name> for each category
    """
    if not request.json and \
        ('name' not in request.json or 'url' not in request.json):
        abort(400)
    name, url = request.json['name'], request.json['url']

    # Is there a deployment that's been registered with the specified url
    deployment = Deployment.by_url(url)
    if deployment is None:
        deployment = Deployment(name=name, url=url)
        deployment.save()
    else:
        return jsonify(deployment.as_dict())
    
    # Add the categories
    _post = request.json
    if 'categories' in _post:
        categories = []
        for cat in _post['categories']:
            category = Category(deployment_id=deployment.id,
                                origin_category_id=cat['origin_category_id'],
                                origin_parent_id=cat['origin_parent_id'],
                                title=cat['title'])
            categories.append(category)

        # Save the categories in bulk
        Category.create_all(categories)

    return jsonify(deployment.as_dict())

@app.route('/v1/deployments/<int:deployment_id>/category', methods=['POST'])
def suggest_categories(deployment_id):
    """Given a message/report, suggests the possible categories
    that the message could fall into
    
    :param deployment_id: the id of the deployment
    """
    # if not request.json:
    #     abort(400)
    # # Does the deployment exist
    # deployment = Deployment.by_id(deployment_id)
    # if not deployment:
    #     abort(404)
    # pass

@app.route('/v1/deployments/<int:deployment_id>/messages', methods=['POST'])
def add_message(deployment_id):
    """Adds a new message for the deployment in :deployment_id
    
    The input parameters are:
        message: string
        
    :param deployment_id: the id of the deployment
    """
    if 'origin_message_id' not in request.json and \
        'content' not in request.json:
        abort(400)

    # Does the deployment exist
    deployment = Deployment.by_id(deployment_id)
    if deployment is None:
        abort(404)
    message = Message(deployment_id=deployment_id,
                      origin_message_id=request.json['origin_message_id'],
                      content=request.json['content'])
    message.create()
    return jsonify(message.as_dict())

@app.route('/v1/deployments/<int:deployment_id>/messages/<int:message_id>',
            methods=['DELETE'])
def delete_message(deployment_id, message_id):
    """Deletes the message with the specified :message_id from
    the deployment specified by the ``deployment_id`` parameter
    
    :param deployment_id: the id of the deployment
    :param message_id: the id of the message
    """
    message = db.session.query(Message).\
        filter(Message.deployment_id == deployment_id,
               Message.origin_message_id == message_id)
    if message is None:
        abort(404)
    message.delete()

@app.route('/v1/deployments/<int:deployment_id>/similar', methods=['POST'])
def similar_messages(deployment_id):
    """
    Given text, finds the near duplicate messages. The duplicate messages
    are specific to the deployment specified in ``deployment_id``
    
    input: text
    output: list. made up of tuples of (id, message text).
    [todo: does this only return reports? or unannotated messages, too?
    should be any message for completeness, and then the front-end can decide
    what should be hidden from the user.]
    
    :param deployment_id: the id of the deployment
    """
    pass

@app.route('/v1/deployments/<int:deployment_id>/reports',
           methods=['POST'])
def add_report(deployment_id):
    """Adds a new report to the deployment specified by the ``deployment_id``
    parameter
    
    Input parameters:
        description: string - Description of the report
        categories: array of integers - category ids
    
    :param deployment_id: the id of the deployment
    """
    if Deployment.by_id(deployment_id) is None:
        abort(404)

    errors = {}
    _post = request.json
    # Check for fields
    if 'origin_report_id' not in _post:
        errors['origin_report_id'] = 'The report id is missing'
    if 'title' not in _post:
        errors['title'] = 'The report title is missing'
    if 'description' not in _post:
        errors['description'] = 'The report description is missing'
    if 'categories' not in _post or len( _post['categories']) == 0:
        errors['categories'] = 'The report categories must be specified'

    # Did we encounter any errors?
    if len(errors) > 0:
        app.logger.error("There are some errors in the request %r" % errors)
        abort(400)

    # Does the specified report already exist?
    _report = db.session.query(Report).\
        filter(Report.origin_report_id == _post['origin_report_id'],
               Report.deployment_id == deployment_id).first()

    if not _report is None:
        app.logger.error("The report %s has already been registered" %
            _post['origin_report_id'])
        abort(400)

    # Get the categories
    categories = db.session.query(Category).\
        filter(Category.deployment_id == deployment_id,
               Category.origin_category_id.in_(_post['categories'])).all()
    
    # Have the specified category ids been registered?
    if len(categories) == 0:
        app.logger.error("The specified categories are invalid")
        abort(400)

    report=Report(deployment_id=deployment_id,
                  origin_report_id=_post['origin_report_id'],
                  title=_post['title'],
                  description=_post['description'])
    # Create the report
    report.create()
    
    # Save the report categories
    report_categories = []
    for category in report_categories:
        rc = ReportCategory(report_id=report.id, category_id=category.id)
        report_categories.append(rc)
    ReportCategory.create_all(report_categories)

    return jsonify(report.as_dict())

@app.route('/v1/deployments/<int:deployment_id>/reports/<int:report_id>',
            methods=['DELETE'])
def delete_report(deployment_id, report_id):
    """Deletes the report with the specified ``report_id`` from the
    deployment referenced by the :deployment_id parameter
    
    :param deployment_id: the id of the deployment
    :param report_id: the id of the report
    """
    report = db.session.query(Report).\
        filter(Report.deployment_id == deployment_id,
               Report.origin_report_id == report_id)
    # Does the report exist?
    if report is None:
        abort(404)
    report.delete()

@app.route('/v1/deployments/<int:deployment_id>/reports/<int:report_id>',
           methods=['PATCH'])
def modify_report(deployment_id, report_id):
    """Modifies the report with the specified :report_id. This report
    must belong to the deployment with the specified :deployment_id
    The :report_id is the database ID of the report in the Ushahidi
    deployment
    
    :param deployment_id: the id of the deployment
    :param report_id: the id of the report
    """
    pass

@app.route('/v1/locations', methods=['POST'])
def suggest_locations():
    """
    Suggest locations in a text string. These might be useful keywords for
    annotators to geolocate.
    
    input: full message's text [string]
    output: list. each item is a python dictionary:
        - text : the text for the specific entity [string]
        - indices : tuple of (start [int], end [int]) offset where entity is
          located in given full message
        - confidence : probability from 0-to-1 [float]
    """
    if not request.json and not 'text' in request.json:
        abort(400)

    # Get all entities and only fetch GPE
    entities = Machine.guess_locations(request.json['text'])
    for k,v in entities.iteritems():
        entities[k] = list(v)

    return jsonify({'locations': entities})

@app.route('/v1/entities', methods=['POST'])
def extract_entities():
    """Given some text input, identify - besides location - people,
    organisations and other types of entities within the text"""
    pass
    if not request.json and not 'text' in request.json:
        abort(400)
    
    result = Machine.guess_entities(request.json['text'])
    
    entities = {}
    for key, value in result.iteritems():
        entities[key.lower()] = list(value)
    
    return jsonify({'entities': entities})
    
@app.route('/v1/private_info', methods=['POST'])
def suggest_sensitive_info():
    """
    Suggest personally identifying information (PII) -- such as
    credit card numbers, phone numbers, email, etc -- 
    from a text string. These are useful for annotators to investigate
    and strip before publicly posting information.
    
    input: text,
    input: options
        - custom regex for local phone numbers
        - flags or booleans to specify the type of pii (e.g. phone_only)
    output: list of dictionaries: 
        - word
        - type (e-mail, phone, ID, person name, etc.)
        - indices (start/end offset in text)
        - confidence [todo: is possible?]
    """
    if not request.json and not 'text' in request.json:
        abort(400)
        
    private_info = Machine.guess_private_info(request.json['text'])
    return jsonify({'private_info': private_info})
