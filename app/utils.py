# Définition des fonctions nécessaires à la préparation des données saisies par les utilisateurs
# ainsi qu'à la prédiction des tags à partir de celles-ci

# Chargement des librairies utiles
import pickle
import contractions
import re
from bs4 import BeautifulSoup
import gensim

import spacy

# Chargement des modèles

unsup_model = pickle.load(open('models/unsup_model_fitted.pkl', 'rb'))

try:
    sp = spacy.load("en_core_web_md")
except:
    spacy.cli.download("en_core_web_md")
    sp = spacy.load("en_core_web_md")

# Chargement/définition des variables utiles

stopwords = sp.Defaults.stop_words

list_100_tags = pickle.load(open('utils/list_100_tags.pkl', 'rb'))
list_short_tags = pickle.load(open('utils/list_short_tags.pkl', 'rb'))

id2word = pickle.load(open('utils/id2word_last.pkl', 'rb'))
dict_topics_20_words = pickle.load(open('utils/dict_20_words_last.pkl', 'rb'))

# Définition des fonctions auxilliaires


def keep_text_from_tags(html, tags=('p')):
    """
        But :
            Analyse le contenu HTML et extrait le contenu textuel des balises passées en argument.
        Arguments :
            htmt : document au format HTML.
            tags : list des balises dont l'on veut extraire le contenu textuel.
        Valeur retournée :
            Une chaîne de caractère issue de la jointure des contenus textuels des différentes balises analysées.
    """
    soup = BeautifulSoup(html, "lxml")
    list_text = list()

    for tag in tags:
        list_text.extend([x.get_text() for x in soup.find_all(tag)])

    return ' '.join(list_text)


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
    #pos_tagged_tokens = t2.tag(tokens)

    #nouns = [word for (word, pos_tag) in pos_tagged_tokens if (pos_tag[0] in wordnet_map.keys())]

    return None

# Définition des fonctions principales


def preprocess_from_raw_text(inputs):
    """
        But :
            Procéder au pré-traitement des données brutes issues de la saisie des utilisateurs de l'API
        Argument :
            inputs : texte saisi par l'utilisateur
        Valeur retournée:
            cleaned_inputs : liste de tokens nettoyés
    """
    cleaned_inputs = keep_text_from_tags(inputs)
    cleaned_inputs = str(cleaned_inputs).lower()
    cleaned_inputs = contractions.fix(cleaned_inputs)
    cleaned_inputs = remove_punct(cleaned_inputs)
    cleaned_inputs = [word.text for word in sp(cleaned_inputs)]
    cleaned_inputs = [word for word in cleaned_inputs if not word in stopwords]
    cleaned_inputs = [word for word in cleaned_inputs if (21 >= len(word) > 3 or word in list_short_tags)]
    cleaned_inputs = ' '.join(cleaned_inputs)
    cleaned_inputs = [noun.text for noun in sp(cleaned_inputs).noun_chunks]
    cleaned_inputs = ' '.join(cleaned_inputs)
    cleaned_inputs = [word.lemma_ for word in sp(cleaned_inputs)]

    return cleaned_inputs



def predictions(cleaned_inputs):
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

