nominator=[2.717*10^9];
denominator=[1   3408.26 3008*400.26  0];
transfer=tf(nominator,denominator);
kp3=0.56878;
kd3=kp3*Td;
ki3=kp3/Ti;
