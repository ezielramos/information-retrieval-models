import math
from pathlib import Path

class Build_Model:
    def __init__(self, _path_docs:Path, _path_qry:Path, process_all:bool):
        self.a = 0.4
        self.data = [(file.read_text(), int(file.name[:-4])) for file in _path_docs.iterdir()]
        self.qry = [(file.read_text(), int(file.name[:-4])) for file in _path_qry.iterdir()]
        self.N = len(self.data)
        self.Q = len(self.qry)
        self.Frecuency = {}
        self.frec_t_q = {}
        self.Max_frecuency_by_document = {}
        self.Max_frecuency_by_query = {}
        self.documents_with_term = {} #documents wich contain term i (n_i)
        self.index_term = {} #indexed all terms in corpus (vocabulary)
        self.i = 0           #for indexing terms
        self.proces_data()
        self.M = len(self.index_term)
        self.idf = [0 for _ in range(self.M)]
        self.tf_table = [[0 for _ in range(self.M)] for _ in range(self.N)]
        self.get_tf_table()
        self.get_idf()
        self.doc_weight = [[0 for _ in range(self.M)] for _ in range(self.N)]
        self.q_weight = [[0 for _ in range(self.M)] for _ in range(self.Q)]
        self.Rank_sim = [[0 for _ in range(self.Q)] for _ in range(self.N)]

        self.get_doc_weight()

        if process_all:
            self.get_query_weight()
            self.get_ranking()

    def proces_data(self):
        for (doc, id) in self.data:            
            #..-> FOR EACH DOC ***
            ###########################################################################
            self.Max_frecuency_by_document[id] = 0   #..-> get max frecuency in the document         
            # self.index_doc[id] = self.j                   #..-> indexing docs
            # self.j += 1
            terms = doc.split()
            set_terms  = set(terms)
            differet_terms = list(set_terms)    #..-> listed terms (remove repeated)
            differet_terms.sort()
            ###########################################################################

            for t in differet_terms:
                if(not self.index_term.__contains__(t)):
                    self.index_term[t] = self.i   #..-> listing all corpus terms
                    self.i += 1

                if(not self.documents_with_term.__contains__(self.index_term[t])):    #..-> n_i
                    self.documents_with_term[self.index_term[t]] = 1
                else: self.documents_with_term[self.index_term[t]] += 1

                f = terms.count(t)
                self.Frecuency[(t, id)] = f
                self.Max_frecuency_by_document[id] = max(f, self.Max_frecuency_by_document[id])

    def get_tf_table(self):
        # for id_doc, index_d in self.index_doc.items():
        for dj in range(self.N):
            m = self.Max_frecuency_by_document[dj]
            for t, index_t in self.index_term.items():
                pair = (t, dj)
                if self.Frecuency.__contains__(pair):
                    self.tf_table[dj][index_t] = self.Frecuency[pair]/m

    def get_idf(self):
        for index_t in range(self.M):
            self.idf[index_t] = math.log2(self.N/self.documents_with_term[index_t])

    def get_doc_weight(self):
        for dj in range(self.N):
            for _, index_t in self.index_term.items():
                tf = self.tf_table[dj][index_t]
                _idf = self.idf[index_t]
                self.doc_weight[dj][index_t] = tf*_idf

    def get_query_weight(self):
        for (_q, id) in self.qry:
            text = _q.split()
            self.Max_frecuency_by_query[id] = 0
            for t in self.index_term:
                f = text.count(t)
                self.frec_t_q[(t, id)] = f
                self.Max_frecuency_by_query[id] = max(self.Max_frecuency_by_query[id], f)

        for q in range(self.Q):
            for t, index_t in self.index_term.items():
                _idf = self.idf[index_t]
                _f_i_q = self.frec_t_q[(t,q)]
                _max_f_i_q = self.Max_frecuency_by_query[q]
                self.q_weight[q][index_t] = (self.a+(1-self.a)*(_f_i_q /_max_f_i_q))*_idf

    def get_new_query_weight(self, _q:str):
        q_weight = [0 for _ in range(self.M)]
        # for (_q, id) in self.qry:
        
        text = _q.split()
        Max_frecuency_by_query = 0
        
        for t in self.index_term:
            f = text.count(t)
            # self.frec_t_q[(t, id)] = f
            Max_frecuency_by_query = max(Max_frecuency_by_query, f)

        for t, index_t in self.index_term.items():
            _idf = self.idf[index_t]
            _f_i_q = text.count(t)
            _max_f_i_q = Max_frecuency_by_query
            q_weight[index_t] = (self.a+(1-self.a)*(_f_i_q /_max_f_i_q))*_idf

        return q_weight

    def get_query_ranking(self, q_weight):
        rank = [0 for _ in range(self.N)]
        for d in range(self.N):
            x, y, z = 0, 0, 0
            for t in range(self.M):
                x += self.doc_weight[d][t]*q_weight[t]
                w1 = self.doc_weight[d][t]
                w2 = q_weight[t]
                y += w1*w1
                z += w2*w2
            r = x/(math.sqrt(y)* math.sqrt(z))
            rank[d] = round(r, 10)
        return rank

    def get_ranking(self):
        for d in range(self.N):
            for q in range(self.Q):
                x, y, z = 0, 0, 0
                for t in range(self.M):
                    x += self.doc_weight[d][t]*self.q_weight[q][t]
                    w1 = self.doc_weight[d][t]
                    w2 = self.q_weight[q][t]
                    y += w1*w1
                    z += w2*w2
                r = x/(math.sqrt(y)* math.sqrt(z))
                self.Rank_sim[d][q] = round(r, 10)