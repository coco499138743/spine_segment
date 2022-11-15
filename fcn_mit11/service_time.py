import random
from collections import OrderedDict

import pandas as pd

log = OrderedDict([
    ('service', []),
    ('type1', []),
    ('type2', []),
    ('type3', []),

])

for i in range(5):
    a = random.randint(1, 9)
    b= random.randint(1, 9)
    c= random.randint(1, 9)
    print(a,b,c)
    type1 = random.uniform(0, 1) * 3 + 2.5
    type2 = random.uniform(0, 1) * 3 + 2.5
    type3 = random.uniform(0, 1) * 3 + 2.5
    if a%3==0:
        type1=0
    elif b%2==0:
        type2=0
    elif c%3==0:
        type3=0
    task_tpye=random.randint(1,3)
    log['service'].append('service%s'%(i+1))
    log['type1'].append(round(type1,2))
    log['type2'].append(round(type2,2))
    log['type3'].append(round(type3,2))

pd.DataFrame(log).to_csv('log_service.csv' , index=False)
