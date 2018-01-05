import logging

import en_vectors_web_lg
import numpy as np
from flask import Flask, jsonify, request, g

from shared.models import KerasModel


logger = logging.getLogger(__name__)


def get_nlp():
    nlp = getattr(g, '_nlp', None)
    if nlp is None:
        nlp = g._nlp = en_vectors_web_lg.load()
    return nlp


def score_text():
    text = "disgusting smelly and totally bad"  # request.json['text']
    nlp = get_nlp()
    model = KerasModel(
        nlp, 'food_text_all',
        max_words_in_sentence=200,
        epochs=5)
    model.load()
    samples = np.array([text], dtype='object')
    prediction = model.predict_proba(samples)[:, 1][0]
    return jsonify(score=float(prediction))


def create_app(environment=None):
    app = Flask(__name__)
    app.config['ENVIRONMENT'] = environment

    init_logging(app.config)

    app.add_url_rule('/', 'test', score_text)
    app.add_url_rule('/score', 'score_text', score_text, methods=['POST'])

    return app


def init_logging(config):
    logging.basicConfig(format="%(levelname)-7s %(name)-8s %(message)s")
    logger = logging.getLogger()
    # TODO - set INFO level depending on config
    logger.setLevel(logging.DEBUG)


if __name__ == '__main__':
    app = create_app()
    app.run('0.0.0.0', use_reloader=True, use_debugger=True, threaded=True)
