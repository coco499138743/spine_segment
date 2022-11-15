import random
from collections import OrderedDict

import pandas as pd

log = OrderedDict([
    ('interval time', []),
    ('arrival time', []),
    ('deadline', []),
    ('task type', []),

])
for i in range(1000):
    interval_time= random.uniform(0, 1)+0.5
    if i == 0:
        arr_time=interval_time
    else:
        arr_time=arr_time+interval_time
    dead_line=arr_time+random.uniform(0, 1)*2+3
    task_tpye=random.randint(1,3)
    log['interval time'].append(interval_time)
    log['arrival time'].append(arr_time)
    log['deadline'].append(dead_line)
    log['task type'].append(task_tpye)

pd.DataFrame(log).to_csv('log.csv' , index=False)
