# Définition des fonctions nécessaires à la préparation des données saisies par les utilisateurs
# ainsi qu'à la prédiction des tags à partir de celles-ci

# Chargement des librairies utiles
import pickle
import contractions
import re
import pandas as pd

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet, brown
from nltk.stem import WordNetLemmatizer

# Chargement des modèles

bert = pickle.load(open('models/bert.pkl', 'rb'))

sup_model = pickle.load(open('models/sup_model_fitted.pkl', 'rb'))
unsup_model = pickle.load(open('models/unsup_model_fitted.pkl', 'rb'))

mlb = pickle.load(open('models/mlb.pkl', 'rb'))

# Chargement/définition des variables utiles

stopwords = set(stopwords.words('english'))

wordnet_map = {"N": wordnet.NOUN}
train_sents = brown.tagged_sents(categories='learned')
t0 = nltk.DefaultTagger('NN')
t1 = nltk.UnigramTagger(train_sents, backoff=t0)
t2 = nltk.BigramTagger(train_sents, backoff=t1)

lemmatizer = WordNetLemmatizer()

list_100_tags = pickle.load(open('utils/list_100_tags.pkl', 'rb'))
list_short_tags = pickle.load(open('utils/list_short_tags.pkl', 'rb'))

id2word = pickle.load(open('utils/id2word.pkl', 'rb'))
dict_topics_20_words = pickle.load(open('utils/dict_topics_20_words.pkl', 'rb'))

# Définition des fonctions auxilliaires


def remove_punct(text):
    """"
        But :
            Supprimer les caractères spéciaux ainsi que la ponctuation,
            les caractères non-ASCII à l'exception de c++ et c#
        Arguments :
            text : texte à nettoyer
        Valeur retournée :
            Texte nettoyé
    """

    pattern_1 = re.compile(r'[^\w]|[\d_]')
    pattern_2 = re.compile(r'[^\x00-\x7f]')

    exceptions = ['c++', 'c#']

    text_splitted = text.split()

    try:
        text_splitted = [re.sub(pattern_1, "", text) for text in text_splitted if not text in exceptions]
        text_splitted = [re.sub(pattern_2, "", text) for text in text_splitted]
    except TypeError:
        return text

    return " ".join(text_splitted)


def pos_tag_nouns(tokens):
    """
        But :

            Filtrer les tokens de type "nom"

        Arguments :

            tokens : tokens à analyser.

        Valeur retournée :

            Liste des tokens de type "nom"
    """
    pos_tagged_tokens = t2.tag(tokens)

    nouns = [word for (word, pos_tag) in pos_tagged_tokens if (pos_tag[0] in wordnet_map.keys())]

    return nouns

# Définition des fonctions principales


def unsup_preprocess_from_raw_text(inputs):
    """
        But :
            Procéder au pré-traitement des données brutes issues de la saisie des utilisateurs de l'API
        Argument :
            inputs : texte saisi par l'utilisateur
        Valeur retournée:
            cleaned_inputs : liste de tokens nettoyés
    """
    cleaned_inputs = str(inputs).lower()
    cleaned_inputs = contractions.fix(cleaned_inputs)
    cleaned_inputs = remove_punct(cleaned_inputs)
    cleaned_inputs = word_tokenize(cleaned_inputs)
    cleaned_inputs = [word for word in cleaned_inputs if not word in stopwords]
    cleaned_inputs = [word for word in cleaned_inputs if (len(word) > 3 or word in list_short_tags)]
    cleaned_inputs = [word for word in cleaned_inputs if word in list_100_tags]
    cleaned_inputs = pos_tag_nouns(cleaned_inputs)
    cleaned_inputs = [lemmatizer.lemmatize(word) for word in cleaned_inputs]
    return cleaned_inputs


def sup_preprocess_from_raw_text(inputs):

    cleaned_inputs = unsup_preprocess_from_raw_text(inputs)
    encoded_inputs = bert.encode(cleaned_inputs)

    return encoded_inputs


def unsup_predictions(cleaned_inputs):
    """
        But:
            Prédire les tags à partir du modèle non-supervisé en considérant tous les thèmes.

        Argument:
            list_of_tokens : liste de tokens issus du nettoyage de la saisie de l'utilisateur

        Valeur retournée:
            relevant_tags : liste des tags prédits
    """
    bowed_text = id2word.doc2bow(cleaned_inputs)
    topics = unsup_model.get_document_topics(bowed_text)

    list_keywords = list()
    for topic_id, topic_prob in topics:
        list_keywords.extend([[round(x[0] * topic_prob, 2), x[1]] for x in dict_topics_20_words[topic_id]])

    potential_tags = sorted(list_keywords, key=lambda i: i[0], reverse=True)[:20]
    potential_tags = [x[1] for x in potential_tags]

    relevant_tags = list(set([tag for tag in potential_tags if tag in cleaned_inputs]))

    return relevant_tags


def sup_predictions(bert_encoded_input):
    """
        But:
            Prédire les tags à partir du modèle supervisé

        Argument:
            bert_encoded_input : vecteur généré par le modèle BERT à partir de liste de tokens issue
                                 du nettoyage de la saisie de l'utilisateur

        Valeur retournée:
            relevant_tags : liste des tags prédits
    """
    sup_features = pd.DataFrame(bert_encoded_input.reshape(1, -1))
    y_pred = sup_model.predict(sup_features)
    tags_pred = mlb.inverse_transform(y_pred)
    tags_pred = list(tags_pred[0])

    return tags_pred
