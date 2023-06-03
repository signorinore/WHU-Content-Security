# successful version

import csv
import numpy as np
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


if __name__ == '__main__':
    stopwords = ['very', 'ourselves', 'am', 'doesn', 'through', 'me', 'against', 'up', 'just', 'her', 'ours',
                'couldn', 'because', 'is', 'isn', 'it', 'only', 'in', 'such', 'too', 'mustn', 'under', 'their',
                'if', 'to', 'my', 'himself', 'after', 'why', 'while', 'can', 'each', 'itself', 'his', 'all', 'once',
                'herself', 'more', 'our', 'they', 'hasn', 'on', 'ma', 'them', 'its', 'where', 'did', 'll', 'you',
                'didn', 'nor', 'as', 'now', 'before', 'those', 'yours', 'from', 'who', 'was', 'm', 'been', 'will',
                'into', 'same', 'how', 'some', 'of', 'out', 'with', 's', 'being', 't', 'mightn', 'she', 'again', 'be',
                'by', 'shan', 'have', 'yourselves', 'needn', 'and', 'are', 'o', 'these', 'further', 'most', 'yourself',
                'having', 'aren', 'here', 'he', 'were', 'but', 'this', 'myself', 'own', 'we', 'so', 'i', 'does', 'both',
                'when', 'between', 'd', 'had', 'the', 'y', 'has', 'down', 'off', 'than', 'haven', 'whom', 'wouldn',
                'should', 've', 'over', 'themselves', 'few', 'then', 'hadn', 'what', 'until', 'won', 'no', 'about',
                'any', 'that', 'for', 'shouldn', 'don', 'do', 'there', 'doing', 'an', 'or', 'ain', 'hers', 'wasn',
                'weren', 'above', 'a', 'at', 'your', 'theirs', 'below', 'other', 'not', 're', 'him', 'during', 'which']
    with open("JobDataAnalyst.csv", "rt", encoding="utf-8") as file:
        reader = csv.reader(file)
        column = [row[3] for row in reader]

        wordlist = ''
        test = []
        for i in range(0, len(column)):
            token = word_tokenize(column[i])
            content = ' '.join(token)
            wordlist = wordlist + content
        wordlist2 = [wordlist]
        print(wordlist)

    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(wordlist2))
    words = vectorizer.get_feature_names()
    weight = tfidf.toarray()

    i = 0
    n = 20  # 前20位
    num = 0  # timer
    for (title, w) in zip(wordlist2, weight):
        loc = np.argsort(-w)
        while num < n:
            if words[loc[i]] not in stopwords:
                print('•{}: {} {}'.format(str(num + 1), words[loc[i]], w[loc[i]]))
                num += 1
                i += 1
            else:
                i += 1

