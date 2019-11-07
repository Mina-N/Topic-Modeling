import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from collections import Counter
#nltk.download('wordnet')      #download if using this module for the first time


from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
#nltk.download('stopwords')    #download if using this module for the first time
from bs4 import BeautifulSoup


#For Gensim
import gensim
import string
from gensim import corpora
from gensim.test.utils import datapath
from gensim.models import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from nltk.tokenize import word_tokenize

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
from nltk import pos_tag

import pandas as pd
import spacy


def main():

    def sent_to_words(sentences):
        for sentence in sentences:
            yield (gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

    def remove_stopwords(texts):
        return [[word for word in gensim.utils.simple_preprocess(str(doc)) if word not in stopwords] for doc in texts]

    def make_bigrams(texts):
        return [bigram_mod[doc] for doc in texts]

    def make_trigrams(texts):
        return [trigram_mod[bigram_mod[doc]] for doc in texts]

    def lemmatization(texts, allowed_postags):
        """https://spacy.io/api/annotation"""
        texts_out = []
        for sent in texts:
            doc = nlp.pipe(" ".join(sent))
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return texts_out

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
                    sent_topics_df = sent_topics_df.append(
                        pd.Series([int(topic_num), round(prop_topic, 4), topic_keywords]), ignore_index=True)
                else:
                    break

        sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']
        return (sent_topics_df)

    def clean(document):
        # Remove unwanted tags using BeautifulSoup
        soup = BeautifulSoup(document, features="html.parser")
        document = soup.get_text()

        # Remove stopwords
        stopwordremoval = " ".join([i for i in document.lower().split() if i not in stopwords])

        # Remove punctuation
        punctuationremoval = ''.join(ch for ch in stopwordremoval if ch not in exclude)

        # Lemmatize Option 1, which runs extremely slowly
        #pos_tags = pos_tag(word_tokenize(punctuationremoval))
        #for tag in pos_tags:
        #    print(tag)
        #    wrd, tg = tag[0], tag[1]
        #    if (tg.startswith('NN')): #noun
        #        normalized.append(lemma.lemmatize(wrd))
        #    elif (tg.startswith('VB')): #verb
        #        normalized.append(lemma.lemmatize(wrd))
        #    elif (tg.startswith('JJ')): #adjective
        #        normalized.append(lemma.lemmatize(wrd))
        #    elif (tg.startswith('RB')): #adverb
        #        normalized.append(lemma.lemmatize(wrd))
        #normalized = " ".join(word for word in normalized)

        # Lemmatize Option 2, which runs quickly
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
        # To only extract 2015 texts, uncomment the last part of the next line
        if (df.loc[i[0], "is_event"] == 0): # & (str(df.loc[i[0], "time_published"]).split("-")[0] == "2015")):
            compileddoc.append(df.loc[i[0], "text"])
            articleNumMapping.append(int(df.loc[i[0], "article_id"]))
            unexcludedText += 1
        else:
            excludedText += 1

    articleNumMapping = pd.Int64Index(articleNumMapping)
    print("Number of isEvent == 1: " + str(excludedText))
    print("Unexcluded Text: " + str(unexcludedText))
    print('\nInformation stored in compileddoc list.')
    stopwords = set(nltk.corpus.stopwords.words('german'))
    exclude = set(string.punctuation)
    exclude.add("–")
    lemma = WordNetLemmatizer()

    #data_words = list(sent_to_words(compileddoc))
    #print("Data words:\n")
    #print(data_words[:1])

    # Build the bigram and trigram models
    #bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)  # higher threshold fewer phrases.
    #trigram = gensim.models.Phrases(bigram[data_words], threshold=100)

    # Faster way to get a sentence clubbed as a trigram/bigram
    #bigram_mod = gensim.models.phrases.Phraser(bigram)
    #trigram_mod = gensim.models.phrases.Phraser(trigram)

    # Remove Stop Words
    #data_words_nostops = remove_stopwords(data_words)

    # Form Bigrams
    #data_words_bigrams = make_bigrams(data_words_nostops)

    # Initialize spacy 'de' model, keeping only tagger component (for efficiency)
    # python3 -m spacy download german model
    #nlp = spacy.load('de', disable=['parser', 'ner'])

    # Do lemmatization keeping only noun, adj, vb, adv
    #data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

    #print("Data Lemmatized:\n")
    #print(data_lemmatized[:1])

    final_doc = [clean(document).split() for document in compileddoc]
    print('\nFiles in compileddoc list cleaned.')
    dictionary = corpora.Dictionary(final_doc)
    DT_matrix = [dictionary.doc2bow(doc) for doc in final_doc]
    print('\nDT_matrix successfully created.')

    # 6 core machine
    lda_model = gensim.models.LdaMulticore(workers=5, corpus=DT_matrix, num_topics=10, id2word=dictionary)
    print('\nModel trained.')
    print(lda_model.print_topics())

    # Compute Perplexity
    print('\nPerplexity: ', lda_model.log_perplexity(DT_matrix))  # The lower, the better

    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(model=lda_model, texts=final_doc, dictionary=dictionary, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', coherence_lda)

    lda_model.save("model_10_all")
    print('\nModel saved.')

    df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model, corpus=DT_matrix, texts=compileddoc)
    print('\nformat_topics_sentences finished executing.')
    print(df_topic_sents_keywords.head(10))

    df_topic_sents_keywords.insert(0, 'Article_ID', articleNumMapping)
    print('\nSet index for data frame.')

    # Format
    # df_dominant_topic = df_topic_sents_keywords.reset_index()
    df_topic_sents_keywords.columns = ['Article_ID', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords']  # , 'Text']
    print('\nData frame successfully created.')
    print(df_topic_sents_keywords.head(10))

    df_topic_sents_keywords.to_csv('all_articles_topics_10.csv')
    print('\nData frame written to csv file.')

    # Visualization of topic dispersion - graph is only useful when number of topics is small (< 20)
    #vis = pyLDAvis.gensim.prepare(lda_model, DT_matrix, dictionary)
    #pyLDAvis.save_html(vis, 'LDA_Visualization_20.html')
    #print('\nHTML file saved.')


if __name__== "__main__":
    main()