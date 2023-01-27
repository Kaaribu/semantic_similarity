import spacy

nlp = spacy.load('en_core_web_md')

# Code extract #!:
word1 = nlp("cat")
word2 = nlp("monkey")
word3 = nlp("banana")
print(word1.similarity(word2))
print(word3.similarity(word2))
print(word3.similarity(word1))

print("")
# Code extract #2:
tokens = nlp('cat apple monkey banana')

for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))

print("")
# Code extract #3:
sentence_to_compare = "Why is my cat on the car"
sentences = ["where did my dog go",
             "Hello, there is my car",
             "I\'ve lost my car in my car",
             "I\'d like my boat back",
             "I will name my dog Diana"]

model_sentence = nlp(sentence_to_compare)

for sentence in sentences:
    similarity = nlp(sentence).similarity(model_sentence)
    print(sentence + " - ", similarity)

# Write a note about what you found interesting about the similarities
# between cat, monkey and banana and think of an example of your own.

'''
One interesting thing to note is that the similarity score between the words "cat" and "monkey" is lower than the 
similarity score between "banana" and "monkey". This suggests that the model sees "banana" as being more semantically 
similar to "monkey" than "cat".

Additionally, the similarity score between "cat" and "banana" is lower than the similarity score between "banana" and
"monkey", which suggest that the model sees "cat" as being less semantically similar to "banana" than "monkey"
'''

'''
An example of my own would be to find the similarity between "dog" and "cat" and "banana", which is likely to be low 
between "dog" and "banana" since they are from different domains and unlikely to have any semantic similarity.
'''
print("")

word1 = nlp("dog")
word2 = nlp("cat")
word3 = nlp("banana")
print(word1.similarity(word2))
print(word3.similarity(word2))
print(word3.similarity(word1))

# Run the example file with the simpler language model ‘en_core_web_sm’ and write a note on what you notice is different
# from the model 'en_core_web_md'.

'''
When running code in the example file with the simpler language model 'en_core_web_sm' as compared to 'en_core_web_md', 
the main difference that I notice is the accuracy of the similarity scores.

Additionally, the 'en_core_web_sm' model has fewer number of parameters, which means it is less complex and less 
accurate than 'en_core_web_md' model. Hence it may not perform as well as the 'en_core_web_md' model on tasks that 
require a deeper understanding of the language.

In summary, the main difference between 'en_core_web_sm' and 'en_core_web_md' is the amount of data and complexity of 
the model, the smaller and simpler 'en_core_web_sm' is less accurate than 'en_core_web_md' in tasks such as similarity 
comparison.
'''
