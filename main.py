import os
import numpy as np
from RVFL import RVFL_train
from gen_ball import gen_balls

directory = './Data'
file_list = os.listdir(directory)
if __name__ == '__main__':

    for file_name in file_list:
        if file_name.endswith(".csv"):
            file_path = os.path.join(directory, file_name)
            print(directory)
            print(file_name)
            file_data = np.loadtxt(file_path, delimiter=',')
            
            m, n = file_data.shape
            for i in range(m):
                if file_data[i, n-1] == 0:
                    file_data[i, n-1] = -1
        
            np.random.seed(0)
            indices = np.random.permutation(m)
            file_data = file_data[indices]
            A_train=file_data[0:int(m*(1-0.30))]
            A_test=file_data[int(m * (1-0.30)):]
            #A, B = GanBall_main(AA_train)
            #A_train = np.vstack((A, B))
        
            m, n = A_train.shape
            indices = np.random.permutation(m)
            A_train = A_train[indices]

            m1, n1 = A_test.shape
            for i in range(m1):
                if A_test[i, n1-1] == -1:
                    A_test[i, n1-1] = 0

            pur = 1 - (0.015 * 5)                      
            num = 2
            A_train = gen_balls(A_train, pur=pur, delbals=num)
            
            Radius=[]
            for i in A_train:
                Radius.append(i[1])
            Center=[]
            for ii in A_train:
                Center.append(ii[0])
            Label=[]
            for iii in A_train:
                Label.append(iii[-1])
            Radius=np.array(Radius)

            Center=np.array(Center)
            Label=np.array(Label)
            
            A_train=np.hstack((Center,Label.reshape(Label.shape[0], 1)))

            m, n = A_train.shape
            for i in range(m):
                if A_train[i, n-1] == -1:
                    A_train[i, n-1] = 0

            C1 = 0.00001
            C2 = 0.00001
            N = 23
            Act = 1

            Test_accuracy, Test_time = RVFL_train(A_train, A_test, C1,C2, N,Act)
            print(Test_accuracy)


        