# Définition des fonctions nécessaires à la préparation des données saisies par les utilisateurs
# ainsi qu'à la prédiction des tags à partir de celles-ci

import pickle

sup_model = pickle.load(open('models/sup_model_fitted.pkl', 'rb'))
unsup_model = pickle.load(open('models/unsup_model_fitted.pkl', 'rb'))


def preprocess_from_raw_text(inputs):

    return None


def unsup_predictions(cleaned_input):

    return None


def sup_predictions(bert_encoded_input):

    return None
