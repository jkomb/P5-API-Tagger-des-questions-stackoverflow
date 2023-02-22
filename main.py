from flask import Flask, request, render_template
from app.utils import unsup_preprocess_from_raw_text, sup_preprocess_from_raw_text, unsup_predictions, sup_predictions


app = Flask(__name__)


@app.route('/', methods=['POST', 'GET'])
def home():
    return render_template('api_page.html')


@app.route('/predict', methods=['POST'])
def predict():

    inputs = request.form.to_dict()
    inputs = list(inputs.values())
    inputs = ' '.join(inputs)

    unsup_features = unsup_preprocess_from_raw_text(inputs)
    sup_features = sup_preprocess_from_raw_text(inputs)

    unsup_tags = unsup_predictions(unsup_features)
    sup_tags = sup_predictions(sup_features)

    final_tags = unsup_tags + sup_tags
    final_tags = list(set(final_tags))
    pred_tags = ', '.join(final_tags)

    if len(unsup_tags) == 0 and len(sup_tags) == 0:

        message_prediction = "Nous sommes navrés, mais nous n'avons pas pu générer de tags... " \
                             "Essayer d'être plus spécifique dans la description de votre problème " \
                             "en incluant par exemple le langage de programmation que vous utilisez, ou la " \
                             "librairie ou le logiciel qui vous donne du fil à retordre !"

    else:

        message_prediction = f"Tags générés : {pred_tags}\n." \

    return render_template('api_page.html', message_prediction=message_prediction)


if __name__ == "__main__":
    app.run(debug=True)
