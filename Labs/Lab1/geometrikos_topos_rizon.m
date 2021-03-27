%Define Transfer function of system 
num = [1];
den =[0.064 0.48 1.2 1];
sys = tf(num,den);

%Root locus plot 
rlocus(sys)

