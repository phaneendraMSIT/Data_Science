# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 03:09:20 2019

@author: phaneendra
"""
# from crypt import methods
from contextlib import redirect_stderr

from flask import Flask, request, render_template, session, redirect, url_for, abort
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField

# App config.
DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)

class ReusableForm(Form):
    gre = TextField('GRE:', validators=[validators.required()])
    toefl = TextField('Toefl',validators=[validators.required()])
    
    @app.route('/', methods=['GET', 'POST'])
    def input_variable():
        form = ReusableForm(request.form)
        if request.method == 'POST':
            print("this is example")
            gre = request.form['greScore']
            print(gre)
            toefl = request.form['toeflScore']
            print(toefl)
        return render_template('GraduateAdmissions.html', form = form)


if __name__ == "__main__":
    app.run(debug=True)