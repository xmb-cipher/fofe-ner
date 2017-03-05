import os
from flask import Flask, render_template,request, json, jsonify


app = Flask(__name__)


@app.route('/', methods = ['GET'])
def homePage():
	print render_template( 'ner-home.html' )
	return render_template( 'ner-home.html' )


@app.route('/', methods = ['POST'] )
def annotate():

	print ' '.join( text.strip() for text in request.form )

	result = jsonify({
	    'text'     : "Ed O'Kelley was the man who shot the man who shot Jesse James.",
	    'entities' : [
	        ['T1', 'PER', [[0, 11]]],
	        ['T2', 'PER', [[20, 23]]],
	        ['T3', 'PER', [[37, 40]]],
	        ['T4', 'PER', [[50, 61]]],
	    ],
	})
	print result
	return result


if __name__== '__main__':
    app.run( '0.0.0.0', 20540 )
