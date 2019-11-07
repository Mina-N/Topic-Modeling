# Topic-Modeling

1. Create a directory named 'data_dta'.

2. Place the rma_articles.dta file into the directory you just created.

3. Create a virtual Python environment to install the necessary modules.<br/>
   For PC:
   
   pip install virtualenv<br/>
   virtualenv [name of virtual environment directory of your choosing]<br/>
   [name of virtual environment directory of your choosing]\Scripts\activate<br/>
       
   Then, install the necessary modules using:<br/>
   pip install ...
   
   For Mac:
   
   pip install virtualenv<br/>
   virtualenv [name of virtual environment directory of your choosing]<br/>
   source [name of virtual environment directory of your choosing]/bin/activate<br/>
   
   Then, install the necessary modules using:<br/>
   pip install ...
   
4. To run lda_gensim.py, which trains an LDA model on a text corpus, simply run<br/>
   python lda_gensim.py
   
   To alter the number of topics that the model is trained to recognize, modify line 149.<br/> 
   To alter the file name that the topic information is written to, modify line 177.
   
5. To run append_topics.py, which loads an LDA model and then writes its information to a file, simply run<br/>
   python append_topics.py
   
   To alter the name of the model being loaded, change line 105.<br/>
   To alter the file name that the topic information is written to, modify line 120.
