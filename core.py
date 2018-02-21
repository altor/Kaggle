import h5py
import numpy as np

def gen_solution(model, scaler, bad_dim, score):
    f = h5py.File('kaggle_lille1_2018_test_v1.save', 'r')
    data = np.array(f['dataset_1'])
    data = np.delete(data, bad_dim, 1)
    data = scaler.transform(data)
    
    output = open('output' + str(score) + '.csv', 'w+')
    output.write("# Id,#Class\n")
    prediction = model.predict(data)
    for i in range(len(prediction)):
        output.write(str(i) + "," + str(int(prediction[i])) + "\n")
