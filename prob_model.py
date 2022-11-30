import math

class Build_BIM:
    
    def __init__(self, docs:list[tuple[str, int]], queries:list[tuple[str, int]], documents_with_term, index_term):
        self.data = docs
        self.qry = queries
        self.N = len(self.data)
        self.Q = len(self.qry)
        self.documents_with_term = documents_with_term
        self.index_term = index_term
        self.Rank_sim = [[0 for _ in range(self.Q)] for _ in range(self.N)]
        self.get_ranking()

    def get_relevance_feedback_ranking(self, query, retrieval_user_relevant):
        rank = [0 for _ in range(self.N)]
        V = len(retrieval_user_relevant)
        _q, _ = self.qry[query]
        set_terms_query = set(_q.split())
        for (doc, id_doc) in self.data:
            set_terms_doc = set(doc.split())
            intersect = set_terms_doc & set_terms_query
            if len(intersect) == 0: continue
            sim = 0
            for t in intersect:
                
                V_t = 0
                for Dr in retrieval_user_relevant:
                    if t in self.data[Dr][0].split(): V_t += 1
                
                dft = self.documents_with_term[self.index_term[t]]  # n_i
                # pt = (self.N + 2*dft)/(3*self.N)    #1/3 + 2dft/3N
                # ut = dft/self.N
                # k = 1

                #relevance feedback modify
                # pt = P_t_input[self.index_term[t]]

                pt2 = (V_t + dft/self.N)/(V + 1)

                ut2 = (dft - V_t + dft/self.N)/(self.N - V + 1)

                fract = (pt2*(1 - ut2))/(ut2*(1 - pt2))

                sim += math.log2(fract)
            
            rank[id_doc] = sim
            
        return rank

    def get_new_query_rank(self, _q):
        rank = [0 for _ in range(self.N)]
        set_terms_query = set(_q.split())
        for (doc, id_doc) in self.data:
            set_terms_doc = set(doc.split())
            intersect = set_terms_doc & set_terms_query
            if len(intersect) == 0: continue
            sim = 0                
            for t in intersect:
                dft = self.documents_with_term[self.index_term[t]]
                pt = (self.N + 2*dft)/(3*self.N)    #1/3 + 2dft/3N
                ut = dft/self.N
                fract = (pt*(1 - ut))/(ut*(1 - pt))
                sim += math.log2(fract)
            rank[id_doc] = sim
        return rank

    def get_ranking(self):
        for (_q, id_q) in self.qry:
            set_terms_query = set(_q.split())
            for (doc, id_doc) in self.data:
                set_terms_doc = set(doc.split())
                intersect = set_terms_doc & set_terms_query
                if len(intersect) == 0: continue
                sim = 0                
                for t in intersect:
                    dft = self.documents_with_term[self.index_term[t]]  #n_i
                    pt = (self.N + 2*dft)/(3*self.N)    #1/3 + 2dft/3N
                    ut = dft/self.N
                    fract = (pt*(1 - ut))/(ut*(1 - pt))
                    sim += math.log2(fract)
                self.Rank_sim[id_doc][id_q] = sim
