from bs4 import BeautifulSoup
import requests

page_link = 'https://en.wikipedia.org/wiki/Google'
page_response = requests.get(page_link, timeout=5)

page_content = BeautifulSoup(page_response.content, "html.parser")
with open('input.txt', 'w', encoding='utf-8') as f:
    f.write(page_content.get_text())

input = page_content.get_text()

import nltk

#TOKENIZATION
#nltk.download('punkt')

stokens = nltk.sent_tokenize(input)
wtokens = nltk.word_tokenize(input)

f= open('sentence.txt','w',encoding='utf-8')
for s in stokens:
    f.write(s+'\n')
    #print(s)

f= open('words.txt','w',encoding='utf-8')
for w in wtokens:
    f.write(w+'\n')

#PARTS OF SPEECH TAGGING
#nltk.download('averaged_perceptron_tagger')
print(nltk.pos_tag(wtokens))
f = open('pos_tagging.txt','w',encoding='utf-8')
f.write(str(nltk.pos_tag(wtokens)))

#STEMMING
from nltk.stem import LancasterStemmer
lStemmer = LancasterStemmer()
print(lStemmer.stem(input))

#LEMMATIZATION
from nltk.stem import WordNetLemmatizer
#nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
print(lemmatizer.lemmatize(input))

#TRIGRAM
#from nltk.util import ngrams
trigram = ngrams(wtokens,3)
for words in trigram:
    print(words)


#NAMED ENTITY RECOGNITION
#nltk.download('maxent_ne_chunker')
#nltk.download('words')
from nltk import wordpunct_tokenize, pos_tag, ne_chunk
print(ne_chunk(pos_tag(wordpunct_tokenize(input))))