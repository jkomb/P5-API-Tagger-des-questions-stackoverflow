from flask import Flask, request, render_template
from utils import unsup_preprocess_from_raw_text, sup_preprocess_from_raw_text, unsup_predictions, sup_predictions


app = Flask(__name__)


@app.route('/')
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

    final_tags = ' '.join(set(unsup_tags+sup_tags))

    if len(unsup_tags) == 0 and len(sup_tags) == 0:

        message_prediction = "Nous sommes navrés, mais nous n'avons pas pu générer de tags...\n" \
                             "Essayer d'être plus spécifique dans la description de votre problème " \
                             "en incluant par exemple le langage de programmation que vous utilisez, ou la " \
                             "librairie ou le logiciel qui vous donne du fil à retordre !"

    else:

        sup_part = 100*round(len(sup_tags)/(len(sup_tags)+len(unsup_tags)), 0)
        unsup_part = len(unsup_tags)/(len(sup_tags)+len(unsup_tags))

        message_prediction = f"Tags générés : {final_tags}\n" \
                             f"\tPart du modèle supervisé dans la prédiction : {sup_part}%\n" \
                             f"\tPart du modèle non-supervisé dans la prédiction : {unsup_part}%\n"

    return render_template('api_page.html', statement=message_prediction)


if __name__ == "__main__":
    app.run(debug=True)
