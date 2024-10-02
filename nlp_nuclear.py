import spacy
import pandas as pd
import transliterate
from deep_translator import GoogleTranslator
import langdetect
import re
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from wordcloud import WordCloud

# Подгружаем файлик с ответами и русскоязычную модель и будем чистить его
nlp = spacy.load("ru_core_news_lg")
df = pd.read_csv('responses.csv', sep=';', header=None, names=['answers'])


# Cначала переводим все ответы на русский язык, для чего используется
# сначал определяем язык каждого ответа, а затем переводим те, что содержат английский
def detect_language(text):
    words = re.findall(r'\b\w+\b', text)
    languages = [langdetect.detect(word) for word in words]
    en_count = languages.count('en')
    if en_count >= 1:
        return 'en'
    else:
        return 'ru'

languages = ['en' if detect_language(answer) == 'en' else 'ru' for answer in df['answers']]

translator = GoogleTranslator(source='en', target='ru')
df['translated_ans'] = df['answers']

for i, language in enumerate(zip(df['answers'], languages)):
    if language == 'en':
        df['translated_ans'][i] = translator.translate( df['translated_ans'][i])

# На случай, если кому-то было лень переключить раскладку обработаем транслит
def recognize_transliteration(text):
  transliterated_text = transliterate.translit(text, 'ru')
  return transliterated_text

df['translated_ans'] = df['translated_ans'].apply(recognize_transliteration)

# Теперь почистим от эмодзи, знаков препинания и прочего лишнего
def remove_punctuation(text):
    return re.sub(r'[^\w\sа-яА-ЯёЁ]+', '', text)

df['translated_ans'] = df['translated_ans'].apply(remove_punctuation)

# Попробуем подсократить длинные фразы и достать из них главное
def extract_main_word(phrase):
    doc = nlp(phrase)
    if len(phrase) > 4:
      # Выделяем корневой элемент в графе зависимостей между словами
      head_word = [token for token in doc if token.dep_ == "ROOT"][0]

      # Если это существительное, то на вский случай получим
      # его 'характеристики' -- описания или дополнения, относящиеся к нему
      # (работа скучная и работа интересная -- сильно отличаются)
      if head_word.pos_ == "NOUN":
        adjectives = [token for token in doc if token.dep_ == "amod" and
                    token.head == head_word]
        characterizing_nouns = [token for token in doc if token.dep_ in
        ["nsubj", "nmod"] and token.head == head_word and token.pos_ == "NOUN"]
        head_word = head_word.text
        if adjectives:
          adjective = adjectives[0].text
          head_word = head_word + " " + adjective
        elif characterizing_nouns:
          characterizing_noun = characterizing_nouns[0].text
          head_word = head_word + " " + characterizing_noun

      # Если это галгол, то могут быть важны наречия при нём
      elif head_word.pos_ == "VERB":
        adverbs = [token for token in doc if token.dep_ == "advmod " and
                    token.head == head_word]
        head_word = head_word.text
        if adverbs:
          head_word = head_word + " " + adverbs[0].text

      # Отдельно обработаем наречие для случая
      # предложений без сказуемых и подлежащих
      elif head_word.pos_ == 'ADV':
        nouns = [token for token in doc if token.pos_ == "NOUN"]
        head_word = head_word.text
        if nouns:
          head_word = head_word + " " + nouns[0].text

      else:
          head_word = head_word.text

    # Если выражение и так короткое, то оставляем
    else:
      head_word = phrase.text

    return head_word

df['translated_ans_main'] = df['translated_ans'].apply(extract_main_word)

# Для подсчёта косинус-меры будет необходимо выделить леммы у всех слов во фразах
def lemmatize_phrase(phrase):
    doc = nlp(phrase)
    lemmatized_words = [token.lemma_ for token in doc]
    return " ".join(lemmatized_words)

# Формируем словарик с группами синонимов
dictionary = defaultdict(list)
threshold = 0.6

# Так как в подобных опросах достаточно велика вероятность
# дубликатов, то для векторизации слов заведём словарь, чтобы 
# не векторизовать дубликаты
vectors = {}

# Получение векторизированного представления слова
def get_vector(word):
  if word not in vectors:
    vectors[word] = nlp(lemmatize_phrase(word)).vector
  return vectors[word]

# Считаем косинус-меру
def get_similarity(word1, word2):
    vector1 = get_vector(word1)
    vector2 = get_vector(word2)
    return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))


# Для обработки всех слов заводим массив с ними и будем удалять
# уже обработанные
all_words = [word.lower() for word in df['translated_ans_main']]

for word in all_words:
    w_lemmatized = lemmatize_phrase(word)
    v1 = get_vector(word)

    # Проверяем: если слово добавлено в словарь с группами синонимов,
    # то это дубликат ранее добавленных слов -- нам он нужен, кладём 
    # по соответсвующему ключу
    if word not in dictionary and word not in [value for dict_val in 
                    dictionary.values() for value in dict_val]:
        dictionary[word] = [word]
    else:
        # Если слова ещё не было, то создаём новый ключ и кладём его
        # по этому ключу
        key = next((key for key, values in dictionary.items() if word in values), None)
        dictionary[key].append(word)

    # Ищем для слова word все синонимичные слова и добавляем по тому же
    # ключу. После добавления удаляем из списка со всеми словами
    similar_words = [w2 for w2 in all_words if get_similarity(word, w2) > threshold]
    for w2 in similar_words:
        key = next((key for key, values in dictionary.items() if word in values), None)
        dictionary[key].append(w2)
        all_words.remove(w2)

# На всякий случай выгружаем полученный словарик в формате json,
# для возможности проведения с ним дальнейших манипуляций
import json

with open('grouped_dictionary.json', 'w', encoding='utf-8') as f:
    json.dump(dictionary, f, ensure_ascii=False, indent=4)

# Ну и наконец формируем картинку для визуализации облака слов
keys = list(dictionary.keys()) # слова, которые будут отображаться
frequencies = {key: len(dictionary[key]) for key in keys} # размер отображаемых слов

wordcloud = WordCloud(width=800, height=500, background_color="white", colormap="magma",
                      relative_scaling=0.6).generate_from_frequencies(frequencies)

plt.figure(figsize=(16, 12))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
