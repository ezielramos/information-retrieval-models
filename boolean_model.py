from pathlib import Path
from nltk.stem.snowball import SnowballStemmer
from random import shuffle

class Build_Boolean_Model:
    
    def __init__(self, _path_docs:Path):
        self.data = [(file.read_text(), int(file.name[:-4])) for file in _path_docs.iterdir()]
        self.stemmer = SnowballStemmer(language='english')
        self.N = len(self.data)
        self.ALL_DOCS = set([x for x in range(self.N)])
        self.documents_with_term:dict[str, list] = {} #documents wich contain term i
        self.documents_with_term_set:dict[str, set] = {}
        self.vocabulary = set()
        self.proces_data()
        for t, ll in self.documents_with_term.items():
            self.documents_with_term_set[t] = set(ll)

    def proces_data(self):
        for doc, id in self.data:
            temp = set(doc.split())
            self.vocabulary = self.vocabulary | temp
            for t in temp:
                if not self.documents_with_term.__contains__(t):
                    self.documents_with_term[t] = [id]
                else: self.documents_with_term[t].append(id)
    
    def get_fnd(self):
        self.terms_list = []
        self.get_terms()
        self.componentes_conj = []
        self.generate_components(0, "")
        self.total_components = len(self.componentes_conj)
        self.num = [x for x in range(self.total_components)] #INDEX COMPONENTS
        shuffle(self.num)
        self.valid_FND = True
        self.fnd = None
        for k in range(1, self.total_components + 1):
            self.eval_sets(k, [], 0)
            if self.fnd != None: break

    def get_terms(self):
        self.experssion = self.experssion.lower()
        self.experssion = self.experssion.replace('(not', '( not')
        self.original = self.experssion
        self.experssion = self.experssion.replace(' or ', ' ')
        self.experssion = self.experssion.replace(' and ', ' ')
        self.experssion = self.experssion.replace(' not ', ' ')
        self.experssion = self.experssion.replace('(', ' ')
        self.experssion = self.experssion.replace(')', ' ')
        length = len(self.experssion)
        self.experssion = self.experssion.replace('  ',' ')
        while length > len(self.experssion):
            length = len(self.experssion)
            self.experssion = self.experssion.replace('  ',' ')
        end = length - 1
        if self.experssion[end] == ' ' : self.experssion = self.experssion[:end]
        if self.experssion[0] == ' ' : self.experssion = self.experssion[1:]
        terms_set = set(self.experssion.split())
        self.terms_list = list(terms_set)

    def generate_components(self, count, current):
        if count == len(self.terms_list): self.componentes_conj.append(current)
        elif count == 0:
            v = self.terms_list[count]
            self.generate_components(count + 1, current + v)
            self.generate_components(count + 1, current + 'not ' + v)
        else:
            v = self.terms_list[count]
            self.generate_components(count + 1, current + ' and ' + v)
            self.generate_components(count + 1, current + ' and not ' + v)

    def evaluate_FND(self, FND:str, eval_original:str, count_terms):
        if not self.valid_FND: return
        if count_terms == len(self.terms_list): self.valid_FND = (eval(FND) == eval(eval_original))
        else:
            v = self.terms_list[count_terms]        
            self.evaluate_FND(FND.replace(v, 'True'), eval_original.replace(v, 'True'), count_terms + 1)
            if not self.valid_FND: return
            self.evaluate_FND(FND.replace(v, 'False'), eval_original.replace(v, 'False'), count_terms + 1)

    def eval_sets(self, k, current:list, pos):
        if self.fnd != None: return
        if len(current) + (len(self.num) - pos) < k: return
        if pos == len(self.num) and len(current) < k: return
        if len(current) == k: 
            current_FND = []
            self.fnd = None
            for index in current: current_FND.append('('+self.componentes_conj[index]+')')
            posible_FND = ' or '.join(current_FND)
            self.valid_FND = True
            self.fnd = posible_FND
            self.evaluate_FND(posible_FND, self.original, 0)
            if not self.valid_FND: self.fnd = None
        else:
            self.eval_sets(k, current + [self.num[pos]], pos + 1)
            self.eval_sets(k, current, pos + 1)

    def get_docs(self, expr:str):
        self.experssion = expr.replace('NOT', ' NOT')  #" x and ( y or w ) "
        self.experssion = self.experssion.replace('(', ' ( ')
        self.experssion = self.experssion.replace(')', ' ) ')
        
        length = len(self.experssion)
        self.experssion = self.experssion.replace('  ',' ')
        while length > len(self.experssion):
            length = len(self.experssion)
            self.experssion = self.experssion.replace('  ',' ')
        end = length - 1
        if self.experssion[end] == ' ' : self.experssion = self.experssion[:end]
        if self.experssion[0] == ' ' : self.experssion = self.experssion[1:]

        #STEMMING
        ss = self.experssion.split()
        ss2 = [self.stemmer.stem(x) for x in ss]
        self.experssion = ' '.join(ss2)
        if self.experssion.startswith('not '): self.experssion = ' ' + self.experssion
        # print(self.experssion)
        # return

        self.original = None
        self.fnd:str = None
        self.get_fnd()
        for tt in self.terms_list:
            if not self.documents_with_term_set.__contains__(tt):
                self.documents_with_term_set[tt]= set()
        fnd_copy:str = self.fnd
        fnd_copy = fnd_copy.replace('(','')
        fnd_copy = fnd_copy.replace(')','')
        OR_expr = fnd_copy.split(' or ') #COMPUT UNION OF all CC
        or_result:set[int] = set()
        or_sets:list[set] = []
        for comp in OR_expr:
            AND_expr = comp.split(' and ')
            count = len(AND_expr)
            and_term = AND_expr[0]
            and_result:set[int] = set()
            if and_term.startswith('not '): 
                and_term = and_term.replace('not ','')
                and_result = self.ALL_DOCS - self.documents_with_term_set[and_term]
            else: and_result = self.documents_with_term_set[and_term]
            for i in range(1, count):
                and_term = AND_expr[i]                
                if and_term.startswith('not '):
                    and_term = and_term.replace('not ','')
                    and_result = and_result & (self.ALL_DOCS - self.documents_with_term_set[and_term])
                else: and_result = and_result & self.documents_with_term_set[and_term]
            or_sets.append(and_result)
        or_result = set()
        for item in or_sets: or_result = or_result|item
        return or_result
        

