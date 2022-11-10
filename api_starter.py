from flask import Flask, jsonify

import esg_service

app = Flask(__name__)


@app.route("/esg")
def hello_world():
    return jsonify(esg_service.get_esg_response())

