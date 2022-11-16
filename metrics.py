
def precision(RR, Rec):
    'recuperados relevantes/recuperados'
    return RR/Rec

def recobrado(RR, Rel):
    'recuperados relevantes/relevantes'
    if RR == 0: return 0
    return RR/Rel

def medidaF(P, R, beta):
    ''' beta = 1: Igual peso para precision y recobrado (F=F1)
        beta > 1: Mayor peso a la Precision
        beta < 1: Mayor peso al  Recobrado
    '''
    if P==0 and R==0: return 0
    return ((1 + beta*beta)*P*R)/(beta*beta*P + R)

def medidaF1(P, R):
    if P==0 and R==0: return 0
    return (2*P*R)/(P + R)