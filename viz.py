import numpy as np
import h5py
import matplotlib.pyplot as plt

train_file = h5py.File('kaggle_lille1_2018_train_v1.save', 'r')
data = np.array(train_file['dataset_1'])
target = np.array(train_file['labels'])


# data2 = np.delete(a, np.arange(4004)[data[0] == 0], 1)


plt.imshow(np.vstack((data[0:500], data[9000:9500])))
plt.show()

# ok = []
# for dim in np.arange(4004)[data[0] < 0.009]:
#     b = True
#     for x in data:
#         if(x[dim] > 0.009):
#             b = False
#             break
#     ok.append(dim)
#     print((dim, b))



l = np.array([(data.T)[i].var() for i in range(4004)])
l[l<0.000001]

bad_dim = []
for dim in np.arange(4004)[data[0] < 0.009]:
    (data.T)[dim].mean()
    
    b = True
    for x in data:
        if(x[dim] > 0.009):
            b = False
            break
    bad_dim.append(dim)
    print((dim, b))
