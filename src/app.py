import logging

from flask import Flask, jsonify, request


logger = logging.getLogger(__name__)


def score_text():
    text = request.json['text']
    prediction = 0.5
    return jsonify(score=float(prediction))


def create_app(environment=None):
    app = Flask(__name__)
    app.config['ENVIRONMENT'] = environment

    init_logging(app.config)

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
