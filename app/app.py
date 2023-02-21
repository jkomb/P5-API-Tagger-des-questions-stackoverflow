from flask import Flask, request, render_template
from utils import preprocess_from_raw_text, unsup_predictions, sup_predictions
import pickle

bert=pickle.load(open('models/bert.pickle','rb'))

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('api_page.html')


@app.route('/predict', methods=['POST'])
def predict():

    inputs = request.form.to_dict()
    inputs = list(inputs.values())
    inputs = ' '.join(inputs)

    cleaned_input = preprocess_from_raw_text(inputs)
    bert_encoded_input = bert.encode(cleaned_input)

    unsup_tags = unsup_predictions(cleaned_input)
    sup_tags = sup_predictions(bert_encoded_input)

    final_tags = ' '.join(set(unsup_tags+sup_tags))

    return render_template('api_page.html', prediction_text='Tags générés : {}'.format(final_tags))


if __name__ == "__main__":
    app.run(debug=True)
