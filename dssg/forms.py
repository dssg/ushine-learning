from flask.ext.wtf import Form, TextAreaField, SubmitField, validators, ValidationError
 
   
class MessageForm(Form):
    message = TextAreaField("Message",  [validators.Required("Please enter a message.")])
    submit = SubmitField("Go")
    lucky = SubmitField("I'm feeling Lucky")