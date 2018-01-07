import logging
import os

import en_vectors_web_lg
import numpy as np
from flask import Flask, jsonify, request, g, render_template

from shared.models_store import Store

store = None
logger = logging.getLogger(__name__)
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


def index():
    return app.send_static_file('index.html')


def score_text():
    text = request.json['text']
    model_name = request.json['model']
    model = store.get_model(model_name)
    samples = np.array([text], dtype='object')
    prediction = model.predict_proba(samples)[:, 1][0]
    return jsonify(score=float(prediction))


def get_available_models():
    return jsonify(store.configuration)


def create_store():
    global store
    store_file = os.path.realpath(os.path.join(CURRENT_DIR, '../dist/store.json'))
    nlp = en_vectors_web_lg.load()
    store = Store(nlp, store_file)
    store.load()


def create_app(environment=None):
    app = Flask(__name__)
    app.config['ENVIRONMENT'] = environment
    app.config['TEMPLATES_AUTO_RELOAD'] = True

    init_logging(app.config)
    create_store()
    
    app.add_url_rule('/', 'index', index)
    app.add_url_rule('/api/score', 'score_text', score_text, methods=['POST'])
    app.add_url_rule('/api/models', 'get_models', get_available_models)
    return app


def init_logging(config):
    logging.basicConfig(format="%(levelname)-7s %(name)-8s %(message)s")
    logger = logging.getLogger()
    # TODO - set INFO level depending on config
    logger.setLevel(logging.DEBUG)


if __name__ == '__main__':
    app = create_app()
    app.run('0.0.0.0', use_reloader=True, use_debugger=True, threaded=True)
