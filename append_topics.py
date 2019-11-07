import nltk
import pandas as pd
import string
import gensim
from nltk.stem.wordnet import WordNetLemmatizer
from bs4 import BeautifulSoup
from gensim import corpora

def format_topics_sentences(ldamodel, corpus, texts):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    #contents = pd.Series(texts)
    #sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)

def format_topics_sentences_1(ldamodel, corpus, articleNumMapping):
    # Init output
    topic_num_list = []
    prop_topic_list = []
    topic_keywords_list = []

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                topic_num_list.append(int(topic_num))
                prop_topic_list.append(round(prop_topic, 4))
                topic_keywords_list.append(topic_keywords)
            else:
                break
    data = {'Article_ID': articleNumMapping, 'Dominant_Topic': topic_num_list, 'Topic_Perc_Contrib': prop_topic_list, 'Keywords': topic_keywords_list}
    sent_topics_df = pd.DataFrame.from_dict(data)
    return(sent_topics_df)

def clean(document):
    # Remove unwanted tags using BeautifulSoup
    soup = BeautifulSoup(document, features="html.parser")
    document = soup.get_text()
    # Remove stopwords
    stopwordremoval = " ".join([i for i in document.lower().split() if i not in stopwords])
    # Remove punctuation
    punctuationremoval = ''.join(ch for ch in stopwordremoval if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punctuationremoval.split())
    return normalized

compileddoc = []
columns = ["article_id", "text", "is_event", "time_published"]
reader = pd.read_stata('data_dta/rma_articles.dta', chunksize=100000, columns=columns)
df = pd.DataFrame()
print('\nInformation stored in Stata Reader.')

m = 0
for itm in reader:
    df = df.append(itm)
    print(m)
    m += 1
print('\nInformation stored in Data Frame.')

articleNumMapping = []
excludedText = 0
unexcludedText = 0
for i in df.iterrows():
    if (df.loc[i[0], "is_event"] == 0): #& (str(df.loc[i[0], "time_published"]).split("-")[0] == "2015")):
        compileddoc.append(df.loc[i[0], "text"])
        articleNumMapping.append(int(df.loc[i[0], "article_id"]))
        unexcludedText += 1
    else:
        excludedText += 1

#articleNumMapping = pd.Int64Index(articleNumMapping)
print("Excluded Text: " + str(excludedText))
print("Unexcluded Text: " + str(unexcludedText))
print('\nInformation stored in compileddoc list.')

stopwords = set(nltk.corpus.stopwords.words('german'))
exclude = set(string.punctuation)
exclude.add("–")
lemma = WordNetLemmatizer()
final_doc = [clean(document).split() for document in compileddoc]
print('\nFiles in compileddoc list cleaned.')

dictionary = corpora.Dictionary(final_doc)
DT_matrix = [dictionary.doc2bow(doc) for doc in final_doc]
print('\nDT_matrix successfully created.')

ldamodel = gensim.models.LdaMulticore.load("model_10_all")
print('\nModel successfully loaded.')

df_topic_sents_keywords = format_topics_sentences_1(ldamodel=ldamodel, corpus=DT_matrix, articleNumMapping=articleNumMapping)
print('\nformat_topics_sentences finished executing.')
print(df_topic_sents_keywords.head(10))

#df_topic_sents_keywords.insert(0, 'Article_ID', articleNumMapping)
#print('\nSet index for data frame.')

# Format
#df_topic_sents_keywords.columns = ['Article_ID', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords']#, 'Text']
print('\nData frame successfully created.')
print(df_topic_sents_keywords.head(10))

df_topic_sents_keywords.to_csv('all_articles_topics_10_append_topics.csv')
print('\nData frame written to csv file.')


#df_topic_sents_keywords.to_stata('2015_articles_topics.dta', write_index=True)
#print('\nData frame written to stata file.')