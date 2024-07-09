# EF-GBTSVM: Enhanced Feature Based Granular Ball Twin Support Vector Machine



Please cite the following paper if you are using this code.

A. Quadir, M. Sajid, M. Tanveer, P. N. Suganthan. “Enhanced Feature Based Granular Ball Twin Support Vector Machine”, 27th International Conference on Pattern Recognition (ICPR), 2024. 

The experiments are executed on a computing system possessing Python 3.11 software, an Intel(R) Xeon(R) CPU E5-2697 v4 processor operating at 2.30 GHz with 128-GB Random Access Memory (RAM), and a Windows-11 operating platform.

We have deployed a demo of the 'EF-GBTSVM' model using the 'chess_krvkp' dataset.

The following are the best hyperparameters set with respect to the “chess_krvkp” dataset

Regularization Parameter c_1=0.00001,  c_2= 0.00001, N=23, Act = 1

Description of files:

main.py: This is the main file to run selected algorithms on datasets. In the path variable specificy the path to the folder containing the codes and datasets on which you wish to run the algorithm.

gen_ball.py: Generation of granular balls

RVFL.py: To generate the hidden matrix

TWSVM.py: Solving the optimization problem

For a comprehensive understanding of the experimental setup, please refer to the paper. Should you encounter any bugs or issues, feel free to contact A. Quadir at mscphd2207141002@iiti.ac.in.



           

