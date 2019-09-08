import pandas as pd
import re
import string
from nltk.corpus import stopwords
import nltk

ps = nltk.PorterStemmer()
wn = nltk.WordNetLemmatizer()

stop_words = stopwords.words('english')

def load_data(filename):
    data = pd.read_csv(filename ,encoding='utf-8')
    data.loc[data['class'] == 0, 'category'] = 'hate'
    data.loc[data['class'] == 1, 'category'] = 'offensive'
    data.loc[data['class'] == 2, 'category'] = 'neutral'
    data = data[['class','category','tweet']]
    data.columns = ['labels','category','tweet']
    return data

def clean_doc_mod(doc):
	# remove retweet (RT) any reference to web address 
	# remove usernames starts with '@' and up to 15 characters
	text = re.sub(r'(RT)|(http\S+)|(@(\w{1,15}))','',doc)
	# remove the RT : re-tweet and &amp; == '&'
	text = re.sub(r'(&amp;)','and',text)
	# Convert the text into tokens
	tokens = text.split()
	# covert all words to lowercase
	tokens = [word.lower() for word in tokens]
	# Compile regex to remove all punctuations
	rm_punc = re.compile('[%s]'%re.escape(string.punctuation))
	tokens = [rm_punc.sub('',word) for word in tokens]
	# remove all stop words
	tokens = [ps.stem(word) for word in tokens if not word in stop_words]
	return ' '.join(tokens)
	
def add_to_vocab(tokens,vocab):
	# update vocabulary counter
	vocab.update(tokens)

def save_file(text,filename):
	# create one line per word
	data = '\n'.join(text)
	# open a file to save the vocabulary
	file = open(filename,'w')
	# write the vocab data to file
	file.write(data)
	# close the file
	file.close()
	
def clean_doc(doc):
	# remove any reference to web address 
	# remove usernames starts with '@' and up to 15 characters
	text = re.sub(r'(RT)|(http\S+)|(@(\w{1,15}))','',doc)
	# remove the RT : re-tweet and &amp; == '&'
	text = re.sub(r'(&amp;)','and',text)
	# Convert the text into tokens
	tokens = text.split()
	# covert all words to lowercase
	tokens = [word.lower() for word in tokens]
	# Compile regex to remove all punctuations
	rm_punc = re.compile('[%s]'%re.escape(string.punctuation))
	tokens = [rm_punc.sub('',word) for word in tokens]
	# remove all words with non alphabetical characters 
	tokens = [word for word in tokens if word.isalpha()]
	# remove all stop words
	tokens = [word for word in tokens if not word in stop_words]
	# remove all words that are unit length
	#tokens = [word for word in tokens if len(word)>1]
	return ' '.join(tokens)
	
	
	

