import streamlit as st
import os
from pathlib import Path
from statistics import mean, median
from preprocessdata import ProcessData
from relevance_feedback import retroalimentacion_app
from sri_evaluation import evaluate_metrics_app
from vectorial_model import Build_Model
from boolean_model import Build_Boolean_Model
from prob_model import Build_BIM
from nltk.stem.snowball import SnowballStemmer
import tables
import json

if not os.path.isfile('./datasets/cisi/cisi.all') or not os.path.isfile('./datasets/cranfield/cran.all.1400') or not os.path.isfile('./datasets/lisa/LISA0.001'):
    st.write('DATABASE FILES MISSING!')
    st.error('Error!')
    exit()


_path_write = './datasets/cisi/preprocessed_docs/'
if not os.path.isdir(_path_write):
    os.mkdir(_path_write)
    # # cisi
    _path_read = "./datasets/cisi/cisi.all"
    pp = ProcessData(_path_read)
    pp.processcorpus(_path_write, 0)
    # dictionary to json
    with open("cisi_docs_info.json", "w") as outfile: json.dump(pp.original_docs_index, outfile)

_path_write = './datasets/cisi/preprocessed_qry/'
if not os.path.isdir(_path_write):
    os.mkdir(_path_write)
    # # cisi
    _path_read = "./datasets/cisi/CISI.QRY"
    pp = ProcessData(_path_read)    #.processcorpus(_path_write, 0)
    pp.processcorpus(_path_write, 0)
    # dictionary to json
    with open("cisi_query_info.json", "w") as outfile: json.dump(pp.original_docs_index, outfile)

_path_write = './tables/cisi_VECT.csv'
if not os.path.isfile(_path_write):
    BASE = Path(".\\datasets\\cisi")
    _path_corpus = BASE / "preprocessed_docs"
    _path_query = BASE / "preprocessed_qry" 

    vm = Build_Model(_path_corpus, _path_query)
    pm = Build_BIM(vm.data, vm.qry, vm.documents_with_term, vm.index_term)

    tables.make_table(_path_write, vm.Rank_sim, vm.Q)
    tables.make_table('./tables/cisi_BIM.csv', pm.Rank_sim, pm.Q)



_path_write = './datasets/cranfield/preprocessed_docs/'
if not os.path.isdir(_path_write):
    os.mkdir(_path_write)
    # # cranfield
    _path_read = "./datasets/cranfield/cran.all.1400"
    pp = ProcessData(_path_read)
    pp.processcorpus(_path_write, 0)
    # dictionary to json
    with open("cranfield_docs_info.json", "w") as outfile: json.dump(pp.original_docs_index, outfile)

_path_write = './datasets/cranfield/preprocessed_qry/'
if not os.path.isdir(_path_write):
    os.mkdir(_path_write)
    # # cranfield
    _path_read = "./datasets/cranfield/cran.qry"
    pp = ProcessData(_path_read)
    pp.processcorpus(_path_write, 0)
    # dictionary to json
    with open("cranfield_query_info.json", "w") as outfile: json.dump(pp.original_docs_index, outfile)

_path_write = './tables/cranfield_VECT.csv'
if not os.path.isfile(_path_write):
    BASE = Path(".\\datasets\\cranfield")
    _path_corpus = BASE / "preprocessed_docs"
    _path_query = BASE / "preprocessed_qry" 
    vm = Build_Model(_path_corpus, _path_query)
    pm = Build_BIM(vm.data, vm.qry, vm.documents_with_term, vm.index_term)
    tables.make_table(_path_write, vm.Rank_sim, vm.Q)
    tables.make_table('./tables/cranfield_BIM.csv', pm.Rank_sim, pm.Q)



_path_write = './datasets/lisa/preprocessed_docs/'
if not os.path.isdir(_path_write):
    os.mkdir(_path_write)
    # # lisa

    _path_read1 = "./datasets/lisa/LISA0.001"
    _path_read2 = "./datasets/lisa/LISA0.501"
    _path_read3 = "./datasets/lisa/LISA1.001"
    _path_read4 = "./datasets/lisa/LISA1.501"
    _path_read5 = "./datasets/lisa/LISA2.001"
    _path_read6 = "./datasets/lisa/LISA2.501"
    _path_read7 = "./datasets/lisa/LISA3.001"
    _path_read8 = "./datasets/lisa/LISA3.501"
    _path_read9 = "./datasets/lisa/LISA4.001"
    _path_read10 = "./datasets/lisa/LISA4.501"
    _path_read11 = "./datasets/lisa/LISA5.001"
    _path_read12 = "./datasets/lisa/LISA5.501"
    _path_read13 = "./datasets/lisa/LISA5.627"
    _path_read14 = "./datasets/lisa/LISA5.850"

    cnt = ProcessData(_path_read1).processcorpus(_path_write, 0)
    cnt = ProcessData(_path_read2).processcorpus(_path_write, cnt)
    cnt = ProcessData(_path_read3).processcorpus(_path_write, cnt)
    cnt = ProcessData(_path_read4).processcorpus(_path_write, cnt)
    cnt = ProcessData(_path_read5).processcorpus(_path_write, cnt)
    cnt = ProcessData(_path_read6).processcorpus(_path_write, cnt)
    cnt = ProcessData(_path_read7).processcorpus(_path_write, cnt)
    cnt = ProcessData(_path_read8).processcorpus(_path_write, cnt)
    cnt = ProcessData(_path_read9).processcorpus(_path_write, cnt)
    cnt = ProcessData(_path_read10).processcorpus(_path_write, cnt)
    cnt = ProcessData(_path_read11).processcorpus(_path_write, cnt)
    cnt = ProcessData(_path_read12).processcorpus(_path_write, cnt)
    cnt = ProcessData(_path_read13).processcorpus(_path_write, cnt)
    cnt = ProcessData(_path_read14).processcorpus(_path_write, cnt)

    original_docs_index = {}
    for i in range(cnt): original_docs_index[i] = i + 1
    # dictionary to json
    with open("lisa_docs_info.json", "w") as outfile: json.dump(original_docs_index, outfile)

_path_write = './datasets/lisa/preprocessed_qry/'
if not os.path.isdir(_path_write):
    os.mkdir(_path_write)
    # # lisa
    _path_read = "./datasets/lisa/LISA.QUE"
    cnt = ProcessData(_path_read).processcorpus(_path_write, 0)

    original_docs_index = {}
    for i in range(cnt): original_docs_index[i] = i + 1
    # dictionary to json
    with open("lisa_query_info.json", "w") as outfile: json.dump(original_docs_index, outfile)

_path_write = './tables/lisa_VECT.csv'
if not os.path.isfile(_path_write):
    BASE = Path(".\\datasets\\lisa")
    _path_corpus = BASE / "preprocessed_docs"
    _path_query = BASE / "preprocessed_qry" 
    vm = Build_Model(_path_corpus, _path_query)
    pm = Build_BIM(vm.data, vm.qry, vm.documents_with_term, vm.index_term)
    tables.make_table(_path_write, vm.Rank_sim, vm.Q)
    tables.make_table('./tables/lisa_BIM.csv', pm.Rank_sim, pm.Q)




stemmer = SnowballStemmer(language='english')

PROCESS_STOP_WORDS = ['never', 'whenev', 'eight', 'fifti', 'more', 'him', 'call', 'out', 'anoth', 'these', 'thus', 'had', 'ma', 'els', 
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

def clear(text):
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

def removeShortWords(text):
    ll = text.split(' ')
    result = []
    short_sustantive = ['usa','us','uk','sex','law','aid']
    for item in ll:
        if len(item) > 3 or item in short_sustantive: result.append(item)
    text = ' '.join(result)
    return text

def remove_stopwords( text):
    ll = text.split(' ')
    for item in PROCESS_STOP_WORDS:
        while ll.__contains__(item): ll.remove(item)
    text = ' '.join(ll)
    return text

def stemming( text):
    ll = text.split(' ')
    _len = len(ll)
    for i in range(_len):
        _word = stemmer.stem(ll[i])
        ll[i] = _word   #self.stemmer.stem(ll[i])            
    text = ' '.join(ll)
    return text

st.title('BIENVENIDO AL SISTEMA DE RECUPERACIÓN DE INFORMACIÓN') 
st.header('Escoja la opción deseada:')

option = 0

status = st.radio('', ('Procesar consultas de la colección de datos',
                        'Realizar consulta booleana sobre la colección de datos',
                        'Realizar nueva consulta sobre la colección de datos', 
                        'Mostrar evaluacion de estrategias de retroalimentacion',
                        'Imprimir evaluacion de los SRI',
                        'Imprimir la información de la aplicación'))

if status == 'Procesar consultas de la colección de datos':
    option = 1
elif status == 'Realizar consulta booleana sobre la colección de datos':
    option = 2
elif status == 'Realizar nueva consulta sobre la colección de datos':
    option = 3
elif status == 'Mostrar evaluacion de estrategias de retroalimentacion':
    option = 4
elif status == 'Imprimir evaluacion de los SRI':
    option = 5
elif status == 'Imprimir la información de la aplicación':
    option = 6


if option == 1:
    st.header('Ranking de los documentos')
    st.subheader('Corpus cisi')
    consulta = 'What is information science?  Give definitions where possible.' # query .I 3
    st.write(consulta)
    st.write('consulta procesada = inform scienc definit possibl')
    cisi_table_VECT, cisi_table_BIM = './tables/cisi_VECT.csv', './tables/cisi_BIM.csv'

    Query_Rank_vect = tables.get_table_query(cisi_table_VECT, 2)
    Query_Rank_prob = tables.get_table_query(cisi_table_BIM, 2)
    n = len(Query_Rank_vect)
    
    doc_index_rank = [(Query_Rank_vect[i], i) for i in range(n)]
    doc_index_rank.sort()
    doc_index_rank.reverse()
    result = doc_index_rank[0:5]
    st.write('Ranking vectorial')
    for i in range(5): st.write(f'id documento:{result[i][1]}, similitud:{result[i][0]}')

    doc_index_rank = [(Query_Rank_prob[i], i) for i in range(n)]
    doc_index_rank.sort()
    doc_index_rank.reverse()
    result = doc_index_rank[0:5]
    st.write('Ranking probabilistico')
    for i in range(5): st.write(f'id documento:{result[i][1]}, similitud:{result[i][0]}')


    st.subheader('Corpus cranfield')
    consulta = 'what similarity laws must be obeyed when constructing aeroelastic models of heated high speed aircraft .' # query .I 1
    st.write(consulta)
    st.write('consulta procesada = similar law obey construct aeroelast model heat high speed aircraft')
    cranfield_table_VECT, cranfield_table_BIM = './tables/cranfield_VECT.csv', './tables/cranfield_BIM.csv'
    
    Query_Rank_vect = tables.get_table_query(cranfield_table_VECT, 0)
    Query_Rank_prob = tables.get_table_query(cranfield_table_BIM, 0)
    n = len(Query_Rank_vect)
    
    doc_index_rank = [(Query_Rank_vect[i], i) for i in range(n)]
    doc_index_rank.sort()
    doc_index_rank.reverse()
    result = doc_index_rank[0:5]
    st.write('Ranking vectorial')
    for i in range(5): st.write(f'id documento:{result[i][1]}, similitud:{result[i][0]}')

    doc_index_rank = [(Query_Rank_prob[i], i) for i in range(n)]
    doc_index_rank.sort()
    doc_index_rank.reverse()
    result = doc_index_rank[0:5]
    st.write('Ranking probabilistico')
    for i in range(5): st.write(f'id documento:{result[i][1]}, similitud:{result[i][0]}')


    st.subheader('Corpus lisa')
    consulta = '''I AM INTERESTED IN COMPUTER DOCUMENTATION SYSTEMS FOR CHEMICAL PATENTS.
I WOULD BE PLEASED TO RECEIVE INFORMATION ON EITHER PUBLICALLY AVAILABLE
SYSTEMS OR ON IN-HOUSE SYSTEMS. CHEMISTRY, CHEMICAL, PATENTS.'''
    st.write(consulta)
    st.write('consulta procesada = interest comput document system chemic patent receiv inform public avail system hous system chemistri chemic patent')
    lisa_table_VECT, lisa_table_BIM = './tables/lisa_VECT.csv', './tables/lisa_BIM.csv'
    
    Query_Rank_vect = tables.get_table_query(lisa_table_VECT, 2)
    Query_Rank_prob = tables.get_table_query(lisa_table_BIM, 2)
    n = len(Query_Rank_vect)
    
    doc_index_rank = [(Query_Rank_vect[i], i) for i in range(n)]
    doc_index_rank.sort()
    doc_index_rank.reverse()
    result = doc_index_rank[0:5]
    st.write('Ranking vectorial')
    for i in range(5): st.write(f'id documento:{result[i][1]}, similitud:{result[i][0]}')

    doc_index_rank = [(Query_Rank_prob[i], i) for i in range(n)]
    doc_index_rank.sort()
    doc_index_rank.reverse()
    result = doc_index_rank[0:5]
    st.write('Ranking probabilistico')
    for i in range(5): st.write(f'id documento:{result[i][1]}, similitud:{result[i][0]}')
    st.success('Terminado exitosamente!')

elif option == 2:
    st.subheader('Escriba la consulta booleana deseada similar al ejemplo')
    st.write('Ejemplo = presented AND ( problems OR data )')
    st.write('Si usa más de 4 términos sin contar los operadores booleanos NOT, AND, OR, el procesamiento puede tomar mucho tiempo')
    
    query = st.text_input(' ', ' ')

    BASE = Path(".\\datasets\\cisi")
    _path_corpus = BASE / "preprocessed_docs"
    m_cisi = Build_Boolean_Model(_path_corpus)
    
    BASE = Path(".\\datasets\\cranfield")
    _path_corpus = BASE / "preprocessed_docs"
    m_cranfield = Build_Boolean_Model(_path_corpus)

    if query == ' ': pass
    else:
        qry1 = m_cisi.get_docs(query)
        st.subheader('Forma Normal Disyuntiva')
        st.write(m_cisi.fnd)

        st.subheader('Documentos recuperados del corpus CISI')
        if len(qry1)==0: st.write('Ninguno de los documentos contienen los términos usados')
        else: st.write(qry1)

        qry1 = m_cranfield.get_docs(query)
        st.subheader('Documentos recuperados del corpus CRANFIELD')
        if len(qry1)==0: st.write('Ninguno de los documentos contienen los términos usados')
        else: st.write(qry1)

        st.success('Terminado exitosamente!')

elif option == 3:
    st.subheader('Escriba la consulta deseada')
    query = st.text_input(' ', ' ')
    if query == ' ': pass
    else:
        result = clear(query)
        result = removeShortWords(result)
        result = stemming(result)
        result = remove_stopwords(result)
        if len(result) == 0: 
            st.write('Consulta no valida!')
            st.error('Error!')

        else:            
            text = result.split()
            query_size = len(text)
            BASE = Path(".\\datasets\\cisi")
            _path_corpus = BASE / "preprocessed_docs"
            _path_query = BASE / "preprocessed_qry" 
            
            vm = Build_Model(_path_corpus, _path_query, False)
            pm = Build_BIM(vm.data, vm.qry, vm.documents_with_term, vm.index_term)

            st.subheader('Documentos recuperados del corpus CISI')            
            ocur = 0
            for i in range(query_size):
                tt = text[i]
                if vm.index_term.__contains__(tt): ocur+=1

            if ocur == 0: st.write('Ninguno de los documentos contienen los términos usados')
            else:
                q_weight = vm.get_new_query_weight(result)
                rank_vect = vm.get_query_ranking(q_weight)
                
                index_rank = [(rank_vect[i], i) for i in range(vm.N)]
                index_rank.sort()
                index_rank.reverse()
                st.write('Ranking vectorial')
                for i in range(5): st.write(f'id documento:{index_rank[i][1]}, similitud:{index_rank[i][0]}')

                rank_vect = pm.get_new_query_rank(result)

                index_rank = [(rank_vect[i], i) for i in range(pm.N)]
                index_rank.sort()
                index_rank.reverse()
                st.write('Ranking probabilistico')
                for i in range(5): st.write(f'id documento:{index_rank[i][1]}, similitud:{index_rank[i][0]}')

            ####################################################################################################

            BASE = Path(".\\datasets\\cranfield")
            _path_corpus = BASE / "preprocessed_docs"
            _path_query = BASE / "preprocessed_qry" 
            
            vm = Build_Model(_path_corpus, _path_query, False)
            pm = Build_BIM(vm.data, vm.qry, vm.documents_with_term, vm.index_term)

            st.subheader('Documentos recuperados del corpus CRANFIELD')
            ocur = 0
            for i in range(query_size):
                tt = text[i]
                if vm.index_term.__contains__(tt): ocur+=1

            if ocur == 0: st.write('Ninguno de los documentos contienen los términos usados')
            else:
                q_weight = vm.get_new_query_weight(result)
                rank_vect = vm.get_query_ranking(q_weight)
                
                index_rank = [(rank_vect[i], i) for i in range(vm.N)]
                index_rank.sort()
                index_rank.reverse()
                st.write('Ranking vectorial')
                for i in range(5): st.write(f'id documento:{index_rank[i][1]}, similitud:{index_rank[i][0]}')

                rank_vect = pm.get_new_query_rank(result)

                index_rank = [(rank_vect[i], i) for i in range(pm.N)]
                index_rank.sort()
                index_rank.reverse()
                st.write('Ranking probabilistico')
                for i in range(5): st.write(f'id documento:{index_rank[i][1]}, similitud:{index_rank[i][0]}')

            ##############################################################################################

            BASE = Path(".\\datasets\\lisa")
            _path_corpus = BASE / "preprocessed_docs"
            _path_query = BASE / "preprocessed_qry" 
            
            vm = Build_Model(_path_corpus, _path_query, False)
            pm = Build_BIM(vm.data, vm.qry, vm.documents_with_term, vm.index_term)

            st.subheader('Documentos recuperados del corpus LISA')
            ocur = 0
            for i in range(query_size):
                tt = text[i]
                if vm.index_term.__contains__(tt): ocur+=1

            if ocur == 0: st.write('Ninguno de los documentos contienen los términos usados')
            else:
                q_weight = vm.get_new_query_weight(result)
                rank_vect = vm.get_query_ranking(q_weight)
                
                index_rank = [(rank_vect[i], i) for i in range(vm.N)]
                index_rank.sort()
                index_rank.reverse()
                st.write('Ranking vectorial')
                for i in range(5): st.write(f'id documento:{index_rank[i][1]}, similitud:{index_rank[i][0]}')

                rank_vect = pm.get_new_query_rank(result)

                index_rank = [(rank_vect[i], i) for i in range(pm.N)]
                index_rank.sort()
                index_rank.reverse()
                st.write('Ranking probabilistico')
                for i in range(5): st.write(f'id documento:{index_rank[i][1]}, similitud:{index_rank[i][0]}')
                

    st.success('Terminado exitosamente!')

elif option == 4:
    retroalimentacion_app()
    st.success('Terminado exitosamente!')

elif option == 5:
    evaluate_metrics_app()
    st.success('Terminado exitosamente!')

else:
    st.header('Información de la Aplicación')
    st.write('Sistema de Recuperación de Información v2.1')
    st.write('Copyright © 2022: Eziel Ramos')
