
import spacy
nlp= spacy.load("en_core_web_sm")
nlp2= spacy.load("en_core_web_md")
with open ("data/cien.txt","r") as f:
    text=f.read()

doc= nlp(text)

#imprime cada token del doc
print("-----------------------------")
for token in doc[:10]:
    print (token)


#imprime cada oracion
print("-----------------------------")
for sent in doc.sents:
    print (sent)

#imprime oracion 1
print("-----------------------------")
sentence1= list(doc.sents)[0]
print(sentence1)

print("-----------------------------")
token2=sentence1[5]
print(token2)
#acceder al texto de un token
token2.text
print(token2.left_edge)
print(token2.right_edge)

#reconoce el entity de la palabra , un poco como el tema de la palabra
print(token2.ent_type_)
#b = beggining of an etity, inerr of a entity, o= outside of entity
print(token2.ent_iob_)
#regresa a la forma original known-->known
print(token2.lemma_)
#informacion de estructural de gramatica participios, singular..
print(token2.morph)
print(token2.pos_)
print(token2.dep_)
#dice el lenguaje
print(token2.lang_)


text = "Mike enjoys playing football."
doc2= nlp(text)

print("----------------")
print(doc2)

for token in doc2:
    print(token.text,token.pos_,token.dep_)

#muestra lo de arriba pero como en diagrama
print("----------------")
from spacy import displacy
displacy.render(doc2,style="dep")

for ent in doc.ents:
    print(ent.text,ent.label_)


#----------------------------------------

sentence1 =list(doc.sents)[0]
print(sentence1)

#importa palabras similares
import numpy as np

your_word="hello"

ms = nlp2.vocab.vectors.most_similar(np.asarray([nlp2.vocab.vectors[nlp2.vocab.strings[your_word]]]),n=20)
words=[nlp2.vocab.strings[w] for w in ms[0][0]]
distances=ms[2]
print(words)

# muestra porcentaje d similitud de las palabras, notese que cambiando de modelo cambia el porcentaje
#similitud semantica
doc1 = nlp2("I like salty fries and hanburgers.")
doc2=nlp2("Fast food tastes very good.")

print(doc1, "<->",doc2,doc1.similarity(doc2))




#se pueden añadir reglas
print("------------------------------------------")
text="West Chestertenfierldville was referenced in Mr. Deeds."
doc=nlp(text)
for ent in doc.ents:
    print(ent.text,ent.label_)
print("------------------------------------------")


from spacy.matcher import Matcher
nlp=spacy.load("en_core_web_sm")

matcher=Matcher(nlp.vocab)
pattern=[{"LIKE_EMAIL": True}]
matcher.add("Email_Adress",[pattern])

doc=nlp("this is a email address: dad@gmail.com, fsaas@hotmail.com")
matches=matcher(doc)
print(matches)
print(nlp.vocab[matches[0][20]].text,nlp.vocab[matches[1][0]].text)



