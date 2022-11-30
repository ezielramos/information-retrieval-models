from metric_evaluation import *
import streamlit as st

def evaluate_metrics_app():
    st.header('Promedio de las metricas')
    st.subheader('Corpus CISI')
    
    docs_info, query_info, model_data, relevance, UMBRALL =  cisi_prob
    Q = len(query_info)
    Quey_Rank = [tables.get_table_query(model_data, i) for i in range(Q)]
    RELV_docs = get_relevance(relevance, Q)

    F1_values = []
    medida_F_recobrado_values = []
    medida_F_precision_values = []
    Prec_values = []
    Rec_values = []

    for q in range(Q):
        original_query = query_info[str(q)]
        Prec, Recob, medida_F_recobrado, medida_F_precision, medida_F1 = eval_query_metrics(UMBRALL, docs_info, RELV_docs[original_query], Quey_Rank[q])
        if len(RELV_docs[original_query]) == 0: continue
        F1_values.append(medida_F1)
        medida_F_precision_values.append(medida_F_precision)
        medida_F_recobrado_values.append(medida_F_recobrado)
        Prec_values.append(Prec)
        Rec_values.append(Recob)
    
    st.write('Modelo probabilistico')
    st.write(f'Promedio Precisión: {mean(Prec_values)}')
    st.write(f'Promedio Recobrado: {mean(Rec_values)}')
    st.write(f'Promedio Medida F, a = 1.5: {mean(medida_F_precision_values)}')
    st.write(f'Promedio Medida F, a = 0.5: {mean(medida_F_recobrado_values)}')
    st.write(f'Promedio Medida F1: {mean(F1_values)}')

    docs_info, query_info, model_data, relevance, UMBRALL =  cisi_vect
    Q = len(query_info)
    Quey_Rank = [tables.get_table_query(model_data, i) for i in range(Q)]
    RELV_docs = get_relevance(relevance, Q)

    F1_values = []
    medida_F_recobrado_values = []
    medida_F_precision_values = []
    Prec_values = []
    Rec_values = []

    for q in range(Q):
        original_query = query_info[str(q)]
        Prec, Recob, medida_F_recobrado, medida_F_precision, medida_F1 = eval_query_metrics(UMBRALL, docs_info, RELV_docs[original_query], Quey_Rank[q])
        if len(RELV_docs[original_query]) == 0: continue
        F1_values.append(medida_F1)
        medida_F_precision_values.append(medida_F_precision)
        medida_F_recobrado_values.append(medida_F_recobrado)
        Prec_values.append(Prec)
        Rec_values.append(Recob)
    
    st.write('Modelo vecorial')
    st.write(f'Promedio Precisión: {mean(Prec_values)}')
    st.write(f'Promedio Recobrado: {mean(Rec_values)}')
    st.write(f'Promedio Medida F, a = 1.5: {mean(medida_F_precision_values)}')
    st.write(f'Promedio Medida F, a = 0.5: {mean(medida_F_recobrado_values)}')
    st.write(f'Promedio Medida F1: {mean(F1_values)}')

    ###inference
    docs_info, query_info, model_data, relevance, UMBRALL =  cisi_infer
    Q = len(query_info)
    Quey_Rank = [tables.get_table_query(model_data, i) for i in range(Q)]
    RELV_docs = get_relevance(relevance, Q)

    F1_values = []
    medida_F_recobrado_values = []
    medida_F_precision_values = []
    Prec_values = []
    Rec_values = []

    for q in range(Q):
        original_query = query_info[str(q)]
        Prec, Recob, medida_F_recobrado, medida_F_precision, medida_F1 = eval_query_metrics(UMBRALL, docs_info, RELV_docs[original_query], Quey_Rank[q])
        if len(RELV_docs[original_query]) == 0: continue
        F1_values.append(medida_F1)
        medida_F_precision_values.append(medida_F_precision)
        medida_F_recobrado_values.append(medida_F_recobrado)
        Prec_values.append(Prec)
        Rec_values.append(Recob)
    
    st.write('Modelo red de inferencia')
    st.write(f'Promedio Precisión: {mean(Prec_values)}')
    st.write(f'Promedio Recobrado: {mean(Rec_values)}')
    st.write(f'Promedio Medida F, a = 1.5: {mean(medida_F_precision_values)}')
    st.write(f'Promedio Medida F, a = 0.5: {mean(medida_F_recobrado_values)}')
    st.write(f'Promedio Medida F1: {mean(F1_values)}')
    ###




    st.subheader('Corpus CRANFIELD')

    docs_info, query_info, model_data, relevance, UMBRALL =  cranfield_prob
    Q = len(query_info)
    Quey_Rank = [tables.get_table_query(model_data, i) for i in range(Q)]
    RELV_docs = get_relevance(relevance, Q)

    F1_values = []
    medida_F_recobrado_values = []
    medida_F_precision_values = []
    Prec_values = []
    Rec_values = []

    for q in range(Q):
        original_query = query_info[str(q)]
        Prec, Recob, medida_F_recobrado, medida_F_precision, medida_F1 = eval_query_metrics(UMBRALL, docs_info, RELV_docs[original_query], Quey_Rank[q])
        if len(RELV_docs[original_query]) == 0: continue
        F1_values.append(medida_F1)
        medida_F_precision_values.append(medida_F_precision)
        medida_F_recobrado_values.append(medida_F_recobrado)
        Prec_values.append(Prec)
        Rec_values.append(Recob)
    
    st.write('Modelo probabilistico')
    st.write(f'Promedio Precisión: {mean(Prec_values)}')
    st.write(f'Promedio Recobrado: {mean(Rec_values)}')
    st.write(f'Promedio Medida F, a = 1.5: {mean(medida_F_precision_values)}')
    st.write(f'Promedio Medida F, a = 0.5: {mean(medida_F_recobrado_values)}')
    st.write(f'Promedio Medida F1: {mean(F1_values)}')

    docs_info, query_info, model_data, relevance, UMBRALL =  cranfield_vect
    Q = len(query_info)
    Quey_Rank = [tables.get_table_query(model_data, i) for i in range(Q)]
    RELV_docs = get_relevance(relevance, Q)

    F1_values = []
    medida_F_recobrado_values = []
    medida_F_precision_values = []
    Prec_values = []
    Rec_values = []

    for q in range(Q):
        original_query = query_info[str(q)]
        Prec, Recob, medida_F_recobrado, medida_F_precision, medida_F1 = eval_query_metrics(UMBRALL, docs_info, RELV_docs[original_query], Quey_Rank[q])
        if len(RELV_docs[original_query]) == 0: continue
        F1_values.append(medida_F1)
        medida_F_precision_values.append(medida_F_precision)
        medida_F_recobrado_values.append(medida_F_recobrado)
        Prec_values.append(Prec)
        Rec_values.append(Recob)
    
    st.write('Modelo vectorial')
    st.write(f'Promedio Precisión: {mean(Prec_values)}')
    st.write(f'Promedio Recobrado: {mean(Rec_values)}')
    st.write(f'Promedio Medida F, a = 1.5: {mean(medida_F_precision_values)}')
    st.write(f'Promedio Medida F, a = 0.5: {mean(medida_F_recobrado_values)}')
    st.write(f'Promedio Medida F1: {mean(F1_values)}')


    ####
    docs_info, query_info, model_data, relevance, UMBRALL =  cranfield_infer
    Q = len(query_info)
    Quey_Rank = [tables.get_table_query(model_data, i) for i in range(Q)]
    RELV_docs = get_relevance(relevance, Q)

    F1_values = []
    medida_F_recobrado_values = []
    medida_F_precision_values = []
    Prec_values = []
    Rec_values = []

    for q in range(Q):
        original_query = query_info[str(q)]
        Prec, Recob, medida_F_recobrado, medida_F_precision, medida_F1 = eval_query_metrics(UMBRALL, docs_info, RELV_docs[original_query], Quey_Rank[q])
        if len(RELV_docs[original_query]) == 0: continue
        F1_values.append(medida_F1)
        medida_F_precision_values.append(medida_F_precision)
        medida_F_recobrado_values.append(medida_F_recobrado)
        Prec_values.append(Prec)
        Rec_values.append(Recob)
    
    st.write('Modelo de inferencia')
    st.write(f'Promedio Precisión: {mean(Prec_values)}')
    st.write(f'Promedio Recobrado: {mean(Rec_values)}')
    st.write(f'Promedio Medida F, a = 1.5: {mean(medida_F_precision_values)}')
    st.write(f'Promedio Medida F, a = 0.5: {mean(medida_F_recobrado_values)}')
    st.write(f'Promedio Medida F1: {mean(F1_values)}')
    ####

    st.subheader('Corpus LISA')

    docs_info, query_info, model_data, relevance, UMBRALL =  lisa_prob
    Q = len(query_info)
    Quey_Rank = [tables.get_table_query(model_data, i) for i in range(Q)]
    RELV_docs = get_relevance(relevance, Q)

    F1_values = []
    medida_F_recobrado_values = []
    medida_F_precision_values = []
    Prec_values = []
    Rec_values = []

    for q in range(Q):
        original_query = query_info[str(q)]
        Prec, Recob, medida_F_recobrado, medida_F_precision, medida_F1 = eval_query_metrics(UMBRALL, docs_info, RELV_docs[original_query], Quey_Rank[q])
        if len(RELV_docs[original_query]) == 0: continue
        F1_values.append(medida_F1)
        medida_F_precision_values.append(medida_F_precision)
        medida_F_recobrado_values.append(medida_F_recobrado)
        Prec_values.append(Prec)
        Rec_values.append(Recob)
    
    st.write('Modelo probabilistico')
    st.write(f'Promedio Precisión: {mean(Prec_values)}')
    st.write(f'Promedio Recobrado: {mean(Rec_values)}')
    st.write(f'Promedio Medida F, a = 1.5: {mean(medida_F_precision_values)}')
    st.write(f'Promedio Medida F, a = 0.5: {mean(medida_F_recobrado_values)}')
    st.write(f'Promedio Medida F1: {mean(F1_values)}')

    docs_info, query_info, model_data, relevance, UMBRALL =  lisa_vect
    Q = len(query_info)
    Quey_Rank = [tables.get_table_query(model_data, i) for i in range(Q)]
    RELV_docs = get_relevance(relevance, Q)

    F1_values = []
    medida_F_recobrado_values = []
    medida_F_precision_values = []
    Prec_values = []
    Rec_values = []

    for q in range(Q):
        original_query = query_info[str(q)]
        Prec, Recob, medida_F_recobrado, medida_F_precision, medida_F1 = eval_query_metrics(UMBRALL, docs_info, RELV_docs[original_query], Quey_Rank[q])
        if len(RELV_docs[original_query]) == 0: continue
        F1_values.append(medida_F1)
        medida_F_precision_values.append(medida_F_precision)
        medida_F_recobrado_values.append(medida_F_recobrado)
        Prec_values.append(Prec)
        Rec_values.append(Recob)
    
    st.write('Modelo vectorial')
    st.write(f'Promedio Precisión: {mean(Prec_values)}')
    st.write(f'Promedio Recobrado: {mean(Rec_values)}')
    st.write(f'Promedio Medida F, a = 1.5: {mean(medida_F_precision_values)}')
    st.write(f'Promedio Medida F, a = 0.5: {mean(medida_F_recobrado_values)}')
    st.write(f'Promedio Medida F1: {mean(F1_values)}')

    docs_info, query_info, model_data, relevance, UMBRALL =  lisa_infer
    Q = len(query_info)
    Quey_Rank = [tables.get_table_query(model_data, i) for i in range(Q)]
    RELV_docs = get_relevance(relevance, Q)

    F1_values = []
    medida_F_recobrado_values = []
    medida_F_precision_values = []
    Prec_values = []
    Rec_values = []

    for q in range(Q):
        original_query = query_info[str(q)]
        Prec, Recob, medida_F_recobrado, medida_F_precision, medida_F1 = eval_query_metrics(UMBRALL, docs_info, RELV_docs[original_query], Quey_Rank[q])
        if len(RELV_docs[original_query]) == 0: continue
        F1_values.append(medida_F1)
        medida_F_precision_values.append(medida_F_precision)
        medida_F_recobrado_values.append(medida_F_recobrado)
        Prec_values.append(Prec)
        Rec_values.append(Recob)
    
    st.write('Modelo red de inferencia')
    st.write(f'Promedio Precisión: {mean(Prec_values)}')
    st.write(f'Promedio Recobrado: {mean(Rec_values)}')
    st.write(f'Promedio Medida F, a = 1.5: {mean(medida_F_precision_values)}')
    st.write(f'Promedio Medida F, a = 0.5: {mean(medida_F_recobrado_values)}')
    st.write(f'Promedio Medida F1: {mean(F1_values)}')



