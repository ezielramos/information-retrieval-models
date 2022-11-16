from random import shuffle
from statistics import mean
import tables, json, metrics
import matplotlib.pyplot as plt

# json to dictonary
with open('cisi_docs_info.json') as json_file1: Docs_info_cisi:dict = json.load(json_file1)
with open('cisi_query_info.json') as json_file2: Query_info_cisi:dict = json.load(json_file2)
with open('cranfield_docs_info.json') as json_file3: Docs_info_cranfield:dict = json.load(json_file3)
with open('cranfield_query_info.json') as json_file4: Query_info_cranfield:dict = json.load(json_file4)
with open('lisa_docs_info.json') as json_file5: Docs_info_lisa:dict = json.load(json_file5)
with open('lisa_query_info.json') as json_file6: Query_info_lisa:dict = json.load(json_file6)

def get_relevance(addr, Q):
    # addr = './datasets/cisi/CISI.REL'
    with open(addr, encoding = "utf-8") as f:
        text = f.readlines()
        f.close()
    rel = [[] for _ in range(Q + 1)]
    for line in text:
        temp = line.split()
        q = int(temp[0])
        d = int(temp[1])
        if q <= Q: rel[q].append(d)
    return rel

def eval_query_metrics(umbral, docs_info, relevant_docs, quey_rank):
    original_to_process = {}
    for proc, orig in docs_info.items():
        original_to_process[orig] = int(proc)

    # original_query = query_info[str(q)]

    n = len(quey_rank)
    score_index = [(quey_rank[i], i) for i in range(n)]
    
    retrieval = []    
    for i in range(n):
        if score_index[i][0] >= umbral:
            retrieval.append(score_index[i][1])

    R_Rel, Rel, Retriev, Prec, Recob = 0, 0, 0, 0, 0

    Retriev = len(retrieval)
    if Retriev == 0: return 0, 0, 0, 0, 0
    Rel = len(relevant_docs)

    for d in relevant_docs:
        proc = original_to_process[d]
        if proc in retrieval: R_Rel += 1

    Prec = metrics.precision(R_Rel, Retriev)
    Recob = metrics.recobrado(R_Rel, Rel)
    medida_F_recobrado = metrics.medidaF(Prec, Recob, 0.5)
    medida_F_precision = metrics.medidaF(Prec, Recob, 1.5)
    medida_F1 = metrics.medidaF1(Prec, Recob)

    return Prec, Recob, medida_F_recobrado, medida_F_precision, medida_F1

def plot_list_values(values):
    plt.plot(values)
    plt.show()

def relevant_rank_degree(q, docs_info, query_info, quey_rank, relevant_docs):
    original_to_process = {}
    for proc, orig in docs_info.items():
        original_to_process[orig] = int(proc)

    sim_relevant_q = []

    original_query = query_info[str(q)]
    QUERY_relevant = relevant_docs[original_query]

    for r in QUERY_relevant:
        # if not original_to_process.__contains__(r):
        #     print("missing", r)
        #     exit()
        proc = original_to_process[r]
        rel = quey_rank[proc]
        sim_relevant_q.append(rel)

    return sim_relevant_q


#######################################################################################################################################################
cisi_rel, cisi_table_VECT, cisi_table_BIM = './datasets/cisi/CISI.REL', './tables/cisi_VECT.csv', './tables/cisi_BIM.csv'
cranfield_rel, cranfield_table_VECT, cranfield_table_BIM = './datasets/cranfield/cranqrel', './tables/cranfield_VECT.csv', './tables/cranfield_BIM.csv'
lisa_rel, lisa_table_VECT, lisa_table_BIM = './datasets/lisa/LISA.REL', './tables/lisa_VECT.csv', './tables/lisa_BIM.csv'

probabilistic_cisi_umbrall, vectorial_cisi_umbrall = 4.845881908117159, 0.0431232508
probabilistic_cranfield_umbrall, vectorial_cranfield_umbrall = 11.942422624849623, 0.0665435744
probabilistic_lisa_umbrall, vectorial_lisa_umbrall = 24.46461382909272, 0.0511557004

cisi_vect = Docs_info_cisi, Query_info_cisi, cisi_table_VECT, cisi_rel, vectorial_cisi_umbrall
cisi_prob = Docs_info_cisi, Query_info_cisi, cisi_table_BIM, cisi_rel, probabilistic_cisi_umbrall
cranfield_vect = Docs_info_cranfield, Query_info_cranfield, cranfield_table_VECT, cranfield_rel, vectorial_cranfield_umbrall
cranfield_prob = Docs_info_cranfield, Query_info_cranfield, cranfield_table_BIM, cranfield_rel, probabilistic_cranfield_umbrall
lisa_vect = Docs_info_lisa, Query_info_lisa, lisa_table_VECT, lisa_rel, vectorial_lisa_umbrall
lisa_prob = Docs_info_lisa, Query_info_lisa, lisa_table_BIM, lisa_rel, probabilistic_lisa_umbrall


# docs_info, query_info, model_data, relevance, UMBRALL =  cisi_prob
# Q = len(query_info)
# Quey_Rank = [tables.get_table_query(model_data, i) for i in range(Q)]


# print(Quey_Rank[26][0])
# print(Quey_Rank[26][1])
# print(Quey_Rank[26][2])

# possible_umbral = []
# RELV_docs = get_relevance(relevance, Q)


# for i in range(Q): 
#     temp = relevant_rank_degree(i, docs_info, query_info, Quey_Rank[i], RELV_docs)
#     possible_umbral.extend(temp)

# shuffle(possible_umbral)

# umbrall = 0
# best_F1_sum = 0
# for u in possible_umbral:
#     current_umbrall = u
#     current_F1_sum = 0
#     for q in range(Q):
#         _, _, _, _, F1 = eval_query_metrics(u, q, docs_info, query_info, RELV_docs, Quey_Rank[q])
#         current_F1_sum += F1
#     if current_F1_sum > best_F1_sum:
#         umbrall = current_umbrall
#         best_F1_sum = current_F1_sum
#         print(best_F1_sum, umbrall)

# print("best umbrall for F1 sum :", umbrall)
# print("F1 sum", best_F1_sum)



# # F1_results = []
# # # for q in range(Q):
# q = 26
# original_query = query_info[str(q)]
# _, _, _, _, F1 = eval_query_metrics(UMBRALL, docs_info, RELV_docs[original_query], Quey_Rank[q])
# print(F1)
# if q == 26: 
#     print("original query =", original_query)
#     to_print = [x - 1 for x in RELV_docs[original_query]]
#     print(to_print)
# if len(RELV_docs[q]) == 0: continue
# if len(RELV_docs[original_query]) == 0: continue
# F1_results.append((F1, q))

# F1_results.sort()
# F1_results.reverse()

# print(F1_results[0:5])

# print("lisa - prob - F1 mean", mean(F1_results))

#$cranfield - vect - F1 mean 0.017387771699608912
#$cranfield - prob - F1 mean 0.24027714820326734
#$cisi - prob - F1 mean 0.14496325847655786
#$cisi - vect - F1 mean 0.062418796448700124
#$lisa - vect - F1 mean 0.009870187217430139
#$lisa - prob - F1 mean 0.16026770083996433
