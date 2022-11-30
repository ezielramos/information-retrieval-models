
from metric_evaluation import Docs_info_cisi, eval_query_metrics, probabilistic_cisi_umbrall, vectorial_cisi_umbrall
from vectorial_model import Build_Model
from prob_model import Build_BIM
import matplotlib.pyplot as plt
from pathlib import Path
import streamlit as st

def retroalimentacion_app():
    st.header('Evaluacion de retroalimentacion de la relevancia')
    
    consulta = 'Computerized information retrieval systems.  Computerized indexing systems.'
    
    st.subheader('Consulta:')
    st.write(consulta)
    st.write('CORPUS: CISI')
    
    query = 26
    a, b, c = 1, 0.75, 0.15
    relevance_feedback_iterations = 9

    BASE = Path(".\\datasets\\cisi")
    cisi_path_corpus = BASE / "preprocessed_docs"
    cisi_path_query = BASE / "preprocessed_qry" 

    vm = Build_Model(cisi_path_corpus, cisi_path_query, True)
    # pm = Build_BIM(vm.data, vm.qry, vm.documents_with_term, vm.index_term)

    original_query_weight = vm.q_weight[query]

    # probabilistic_retrieval = set()
    vectorial_retrieval = set()

    for i in range(vm.N):
        if vm.Rank_sim[i][query] >= vectorial_cisi_umbrall:
            vectorial_retrieval.add(i)



    CISI_relevant_index_docs_q26 = set([26, 43, 50, 51, 56, 64, 67, 70, 71, 76, 77, 78, 113, 116, 122, 125, 148,
        149, 158, 173, 174, 175, 179, 186, 189, 190, 211, 240, 256, 257, 258, 308, 313, 314, 318, 320, 325, 326, 328,
        376, 377, 388, 389, 433, 445, 447, 479, 488, 492, 498, 499, 505, 509, 516, 521, 526, 529, 530, 537, 564, 565,
        569, 589, 599, 602, 631, 643, 658, 669, 670, 672, 679, 683, 686, 694, 698, 699, 703, 708, 739, 745, 757, 760,
        768, 772, 775, 780, 789, 795, 804, 816, 819, 823, 824, 829, 868, 893, 1090, 1119, 1123, 1125, 1126, 1129,
        1131, 1138, 1143, 1229, 1254, 1279, 1282, 1297, 1322, 1391, 1418, 1420])



    CISI_relevant_orig_docs_q26 = [x + 1 for x in sorted(CISI_relevant_index_docs_q26)]

    Rank_sim_vect_q26 = [vm.Rank_sim[i][query] for i in range(vm.N)]

    vect_metrics = eval_query_metrics(vectorial_cisi_umbrall, Docs_info_cisi, CISI_relevant_orig_docs_q26, Rank_sim_vect_q26)

    vectorial_feedback_relevance = [vect_metrics]

    def rocchio(q0, a, b, c, Dr, Dn):
        len_Dr = len(Dr)
        len_Dn = len(Dn)

        relevant_centroid = [0 for _ in range(vm.M)]
        not_relevant_centroid = [0 for _ in range(vm.M)]

        for i in range(vm.M):
            for d in Dr: relevant_centroid[i] += vm.doc_weight[d][i]
            for d in Dn: not_relevant_centroid[i] += vm.doc_weight[d][i]

        for i in range(vm.M):
            relevant_centroid[i] = relevant_centroid[i]/len_Dr
            not_relevant_centroid[i] = not_relevant_centroid[i]/len_Dn

        qm = [(a*q0[i] + b*relevant_centroid[i] - c*not_relevant_centroid[i]) for i in range(vm.M)]

        return qm

    query_weight = original_query_weight.copy()

    for _ in range(relevance_feedback_iterations):
        
        user_select_vectorial = CISI_relevant_index_docs_q26 & vectorial_retrieval
    
        Docs = vectorial_retrieval
        Dr = user_select_vectorial
        Dn = Docs - Dr

        query_weight = rocchio(query_weight, a, b, c, Dr, Dn)   #UPDATE VECT QUERY
        
        rank_vect_feedback = vm.get_query_ranking(query_weight)

        vect_metrics = eval_query_metrics(vectorial_cisi_umbrall, Docs_info_cisi, CISI_relevant_orig_docs_q26, rank_vect_feedback)

        vectorial_feedback_relevance.append(vect_metrics)

        vectorial_retrieval.clear()     #UPADATE RETRIEVAL DOCS

        for i in range(vm.N):
            if rank_vect_feedback[i] >= vectorial_cisi_umbrall:
                vectorial_retrieval.add(i)


    st.subheader('Modelo vectorial')
    for i in range(relevance_feedback_iterations):
        st.subheader(f'iteracion: {i+1}')
        st.write(f'Precisión: {vectorial_feedback_relevance[i][0].__round__(4)} , Recobrado : {vectorial_feedback_relevance[i][1].__round__(5)} , Medida F1 : {vectorial_feedback_relevance[i][4].__round__(5)}')

