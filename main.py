from flask import Flask, request, render_template
from app.utils import preprocess_from_raw_text, predictions


app = Flask(__name__)


@app.route('/', methods=['POST', 'GET'])
def home():
    return render_template('api_page.html')


@app.route('/predict', methods=['POST'])
def predict():

    inputs = request.form.to_dict()
    inputs = list(inputs.values())
    inputs = ' '.join(inputs)

    features = preprocess_from_raw_text(inputs)

    tags = predictions(features)
    tags = ' '.join(tags)

    if len(tags) == 0:

        message_prediction = "Nous sommes navrés, mais nous n'avons pas pu générer de tags... " \
                             "Essayer d'être plus spécifique dans la description de votre problème " \
                             "en incluant par exemple le langage de programmation que vous utilisez, ou la " \
                             "librairie ou le logiciel qui vous donne du fil à retordre !"

    else:

        message_prediction = f"Tags générés : {tags}\n." \

    return render_template('api_page.html', message_prediction=message_prediction)


if __name__ == "__main__":
    app.run(debug=True)
