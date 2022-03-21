import numpy as np

# merge any numpy or list of same dim
def mergeNumpyList(npList, metaList):
    npList = npList.tolist() 
    metaList.extend(npList)
    print("*" * 300)
    print(f'new len: {len(metaList)}')
