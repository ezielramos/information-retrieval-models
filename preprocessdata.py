import os
from nltk.stem.snowball import SnowballStemmer

class ProcessData:
    
    def __init__(self, _path_read):
        self.PROCESS_STOP_WORDS = ['never', 'whenev', 'eight', 'fifti', 'more', 'him', 'call', 'out', 'anoth', 'these', 'thus', 'had', 'ma', 'els', 
            'among', "hadn't", 'fifteen', 'quit', 'alon', 'five', 'those', 'anyhow', 'within', "doesn't", 'although', 'wouldn', "'s", 'three', 'yet', 
            'top', 'forti', 'but', 'until', 'between', 'becam', 'an', 'most', 'who', "that'll", "mustn't", 'herself', 'whereupon', 'though', 'either', 
            'that', 'everywher', 'move', 'first', 'not', 'me', 'whom', 'noth', 'thereaft', "wasn't", 'few', 'ever', 've', 'make', 'yourself', 'hereaft', 
            'even', 'by', 'have', "'m", 'due', 'beyond', 'through', 'were', 'next', 'two', 'becaus', 'at', 'wherebi', 'on', 'toward', 'how', 'twenti', 
            'onli', 'ain', 'give', 'then', 'their', 'nowher', 'than', 'against', 'empti', 'with', 'whenc', 'latter', 'and', 'wasn', 'amount', 'themselv', 
            'couldn', 'say', 'howev', 'almost', 'which', 'down', 'whatev', 'nevertheless', 'therefor', 'do', 'various', 'we', 'each', 'my', 'made', 
            'twelv', 'what', 'anyth', 'along', 'mightn', 'sever', 'there', "hasn't", 'own', 'somehow', 'bottom', 'would', 'weren', 'nor', 'the', 'now', 
            'nine', 'so', 'must', 'alreadi', 'one', 'll', 'up', 'hasn', 'mustn', 'whereaft', 'after', 'can', "wouldn't", 'everyth', 'such', 'is', 
            'anyway', 'therebi', 'via', 'should', 'therein', 'this', 'upon', 'thereupon', 'seem', 'doesn', 'throughout', 'ten', 'while', 'over', "mightn't", 
            'could', 'may', 'might', 'take', 'none', 'will', 'herebi', "'d", 'around', 'wherea', 'for', 'becom', 'inde', 'otherwis', 'shan', "aren't", 
            "needn't", 'both', 'itself', 'they', 'shouldn', 'least', 'moreov', 'part', 'as', 'meanwhil', 'some', 'someth', 't', 'name', 'your', 'about', 
            "don't", 'use', 'just', "'m", "shouldn't", 'are', 'myself', 'onc', 'wherein', 'in', 'y', 'same', 'whose', 'go', 'our', 'them', 'still', 'former', 
            "'d", 'get', 'i', 'see', 'or', "n't", 'thru', 'whither', "won't", 'haven', 'without', 'last', 're', 'of', "haven't", "couldn't", 'thenc', 'cannot', 
            'hundr', 'show', 'four', 'everi', "you'r", 'noon', "you'll", 'often', 'mani', 'ani', "'s", 'alway', 'he', 'done', 'wherev', "you'v", 
            'be', 'anyon', 'hereupon', 'under', 'eleven', 'if', 'onto', 'where', 'less', 'his', 'back', 'unless', 'except', 'much', 'enough', 'no', 
            'd', 'sometim', 'o', 'you', 'mine', 'whoever', 's', 'again', 'yourselv', 'across', 'won', 'isn', 'herein', 'has', 'whole', 'ca', 'pleas', 'other', 
            'all', 'from', 'anywher', 'aren', 'needn', 'third', 'her', 'beforehand', 'here', 'don', 'full', "shan't", 'when', 'perhap', 'well', 'neither', 
            'nobodi', "weren't", 'veri', 'whi', 'henc', 'ourselv', 'didn', 'someon', "didn't", 'hadn', "isn't", 'himself', 'sinc', 'somewher', 'dure', 
            'front', 'everyon', 'befor', 'amongst', 'besid', 'it', 'doe', 'rather', 'too', "should'v", 'per', 'keep', 'into', 'to', 'further', 'also', 
            "you'd", 'elsewher', 'm', 'whether', 'abov', 'afterward', 'regard', 'serious', 'side', 'she', 'been', 'am', 'togeth', 'below', 'put', 'off', 
            'realli', 'was', 'did', 'behind', 'six', 'a', 'sixti']
        f = open(_path_read, "r", encoding = "utf-8")
        self.text = f.read()
        f.close()
        self.original_docs_index = {}
        self.text_list = None
        self.W = None
        self.original_words = {}

        #temporary
        code = "\n********************************************\n"
        code1 = ". #\n"
        if self.text.__contains__(code): self.W = [t[8:] for t in self.text.split(code)]
        elif self.text.__contains__(code1): self.W = self.text.split(code1)
        else:
            self.text_list = self.text.split('\n.I ')
            self.W = [t.split('\n.W')[1][1:].split('\n.X\n')[0] for t in self.text_list]


        self.stemmer = SnowballStemmer(language='english')
        
    def clear(self, text):
        to_remove = ['.', '-', ',', '(', ')', '[', ']', '{', '}', '"', ':', ';', 
        "'", '\n', '\t', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9','  ', 
        '?', '!', '$', '-', '_', '<', '>', '~', '/', '|', '*', '+', '%', '&', 
        '\'', '\\', '^', '@', '#', '=', '`']

        for remv in to_remove:
            text = text.replace(remv,' ')
        
        length = len(text)
        text = text.replace('  ',' ')
        while len(text) < length:
            length = len(text)
            text = text.replace('  ',' ')
        text = text.lower()
        if len(text) > 0 and text[0] == ' ': text = text[1:]
        return text

    def removeShortWords(self, text):
        ll = text.split(' ')
        result = []
        short_sustantive = ['usa','us','uk','sex','law','aid']
        for item in ll:
            if len(item) > 3 or item in short_sustantive: result.append(item)
        text = ' '.join(result)
        return text

    def remove_stopwords(self, text):
        ll = text.split(' ')
        for item in self.PROCESS_STOP_WORDS:
            while ll.__contains__(item): ll.remove(item)
        text = ' '.join(ll)
        return text

    def stemming(self, text):
        ll = text.split(' ')
        _len = len(ll)
        for i in range(_len):
            _word = self.stemmer.stem(ll[i])
            self.original_words[_word] = ll[i]
            ll[i] = _word   #self.stemmer.stem(ll[i])            
        text = ' '.join(ll)
        return text

    def processcorpus(self, _path_write, start_index):
        n, j = len(self.W), start_index
        for i in range(n):
            out = _path_write + str(j) + '.txt'            
            result = self.clear(self.W[i])
            result = self.removeShortWords(result)
            result = self.stemming(result)
            result = self.remove_stopwords(result)            
            if len(result) > 0: 
                f = open(out, "w")
                f.write(result)
                self.original_docs_index[j] = i + 1
                f.close()
                j += 1
        return j
