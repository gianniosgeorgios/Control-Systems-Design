
%the system is in dx/dt = Ax + Bu , y= Cx 
%where
A=[0 1 0 0;20.6 0 0 0;0 0 0 1;-0.5 0 0 0];  %A array of system
B=[0;-1;0;0.5];                             %B array of system
C=[1 0 0 0;0 0 1 0];                        %C array of system
D=[0;0];                                    %D array of system

%% PART_A

syms s                                      %s(variable)
Is=s*eye(4);                                %Array sI
ch_eq=det(Is-A);                            %Characyeristic Polynomial




S=[B A*B A*A*B A*A*A*B];   %Controllabillity array 
S_dt=det(S); 
S_inv=inv(S) ;

%Canonical Control Form Transformation 
q=S_inv(end,:);             
T=[q;q*A;q*A*A;q*A*A*A];   %Τransformation Table
T_inv=inv(T);               
Ac=T*A*T_inv;               
Bc=T*B;                     

%Now we choose poles
pole1=-14; 
pole2=-14;
z=0.5;
Wn=3;
pole3=-z*Wn+(Wn*sqrt(1-z^2))*i;
pole4=-z*Wn-(Wn*sqrt(1-z^2))*i;

poles=[pole1 pole2 pole3 pole4]; 

ad=[1764 840 289 31];           %Coefficients of desired characteristic polynomial 
a0=[0 0 -103/5 0];              %Coefficients of characteristic polynomial (Canonical Form)
initial=[-0.2;-0.06;0.01;0.3];  %Ιnitial Conditions

K=(a0-ad)*T;                    %K (gain) so as to move poles 

K1=(ad-a0)*T;                   %K with negative feedback 

%% PARTB
Q=eye(4);                       
R=1;
[P,L,Kr]=care(A,B,Q);           %Riccati Solution using care function 

%% PARTC
[m,n]=ss2tf(A-B*K1,B,C,D);      %Transfer function so as to find error 
K3=K1(3);% 

%% PART_D
syms j1 j2 j3 j4 j5 j6 j7 j8    %Symbolic variablel 

KK=[4 0 0 0;0 0 -1097.9 0;-0.0933 0 0 0;0 0 26.6 0];


%ftiahnoume anatrofodothsh K etsi wste na einai efstathes 
%h mhtra (A-KK) opou KK =K*C
K4=[4 0;0 -1097.9;-0.0933 0;0 26.6];
xar_pol=expand(det(Is-(A-K4*C)));
roots([1 4 6 3.96593 0.99]);
roots([0.99 3.96593 6 4 1]);

%% PART_E

A5=[0 1 0 0;20.9 0 0 0;0 0 0 1;-0.8 0 0 0]; %new array 




