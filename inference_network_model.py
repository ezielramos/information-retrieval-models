import math

class Build_Inference_Network:
    
    def __init__(self, docs:list[tuple[str, int]], queries:list[tuple[str, int]], index_term, tf_data, idf_data):
        self.data = docs
        self.idf = idf_data
        self.tf_table = tf_data
        self.qry = queries
        self.N = len(self.data)
        self.Q = len(self.qry)
        # self.documents_with_term = documents_with_term
        self.index_term = index_term
        self.Rank_sim = [[0 for _ in range(self.Q)] for _ in range(self.N)]
        self.get_ranking()

    def get_new_query_rank(self, _q):
        a = 0.001
        Cj = [0.0 for _ in range(self.N)]
        for d in range(self.N):
            cj = 1.0
            for _, index_t in self.index_term.items():
                tf = self.tf_table[d][index_t]
                cj *= (1 - tf + a)
            Cj[d] = cj
        rank = [0 for _ in range(self.N)]
        set_terms_query = set(_q.split())
        for (doc, id_doc) in self.data:
            set_terms_doc = set(doc.split())
            intersect = set_terms_doc & set_terms_query
            if len(intersect) == 0: continue
            dj_norm = math.sqrt(len(set_terms_doc))                
            P_dj = 1/dj_norm
            _Cj = Cj[id_doc]
            _sum = 0
            for t in intersect:
                id_t = self.index_term[t]
                tf = self.tf_table[id_doc][id_t]
                _idf = self.idf[id_t]
                _sum += tf*_idf*(1/(1 - tf + a))
            rank[id_doc] = round(_Cj*P_dj*_sum, 10)
        return rank

    def get_ranking(self):
        a = 0.001
        Cj = [0 for _ in range(self.N)]
        for d in range(self.N):
            cj = 1
            for _, index_t in self.index_term.items():
                tf = self.tf_table[d][index_t]
                cj *= (1 - tf + a)
            Cj[d] = cj
        for (_q, id_q) in self.qry:
            set_terms_query = set(_q.split())
            for (doc, id_doc) in self.data:
                set_terms_doc = set(doc.split())
                intersect = set_terms_doc & set_terms_query
                if len(intersect) == 0: continue
                dj_norm = math.sqrt(len(set_terms_doc))                
                P_dj = 1/dj_norm
                _Cj = Cj[id_doc]
                _sum = 0
                for t in intersect:
                    id_t = self.index_term[t]
                    tf = self.tf_table[id_doc][id_t]
                    _idf = self.idf[id_t]
                    _sum += tf*_idf*(1/(1 - tf + a))
                self.Rank_sim[id_doc][id_q] = round(_Cj*P_dj*_sum, 10)
