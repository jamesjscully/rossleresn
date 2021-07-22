
function sm_run
figure(1)
clf;
%close all 
clear all;
set(gcf, 'PaperPositionMode','auto','color', 'white');
set(gcf,'PaperPosition',[1.5 3 5 5])
hFig=axes('Position',[0.2 0.4 .6 .6],'Visible','off','Color',[.9 .9 .9],...
         'FontName','times',...
         'FontSize',8,...
         'XColor',[0 0 .0],...
         'YColor',[0 0 .0],...
         'ZColor',[0 0 .0]);
r=92.5;
name_seq_fake='r925_fake.dat';
name_seq_true='r925_true.dat';


shift=00;
flor = @(t,s) [ -10*(s(1)-s(2)); r*s(1)-s(2)-s(1).*s(3)+shift; -2.6666*s(3)+s(1).*s(2)];
options = odeset('RelTol',1e-4,'AbsTol',1e-5,'Events',@events);
options0 = odeset('RelTol',1e-4,'AbsTol',1e-5);
%options = odeset('Events',@events);

[tt,yy]=ode45(flor,[0 500],[0.001,0.,0.],options0);
yy0=yy(length(tt),:);

[tt,yy,tau1,ye1,ie1]=ode45(flor,[0:0.005:12200],yy0,options);
plot3(yy(1:50000,1), yy(1:50000,2), yy(1:50000,3),'Color', [0.65 0.65 0.65])
hold on
plot3(ye1(:,1), ye1(:,2), ye1(:,3),'.','MarkerSize',10,'Color', [0.99 0.0 0.0])
hold on
% k1=length(y(:,1));
% plot3(y(k1-2000:k1,1), y(k1-2000:k1,2), y(k1-2000:k1,3),'Color', [0.0 0.6 0.0],'LineWidth',3)
% hold on

xlabel('x','FontSize',20);
ylabel('y','FontSize',20);
zlabel('z','FontSize',20);
axis on;
axis([-25 25 -30 30 0 55])
axis tight 
view(10,5)

set(gca,'box','off','FontSize',8);
 hFig=axes('Position',[0.2 0.1 .6 .2],'Visible','off','Color',[.9 .9 .9],...
         'FontName','times',...
         'FontSize',8,...
         'XColor',[0 0 .7],...
         'YColor',[0 0 .7],...
         'ZColor',[0 0 .7]);
box on
plot (tt(1:end),yy(1:end,1),'Color','b','LineWidth',1, 'Color', [0.7 0.7 0.7])
hold on
plot (tau1(1:end),ye1(1:end,1),'.','MarkerSize',15,'Color',[255./255  1./255  1./255])
hold on
xlabel('time','FontSize',20);
ylabel('x','FontSize',20);
axis tight
xlim([600 800])
axis on

%b = logical(heaviside(ye1(:,1)'));
length(ye1(:,1))
b = logical(heaviside(ye1(:,1)));
b2=b+1;
lookup_string1 = '10';
b1 = lookup_string1(b + 1)

[TRANS_EST, EMIS_EST] = hmmestimate(b2,b2);
A=TRANS_EST';
An=[A(2,2) A(2,1); A(1,2) A(1,1)]

fid1=fopen(name_seq_true,'w');
for i=1:length(b)
      fprintf(fid1,'%d\n',b(i));
end
st=fclose (fid1);

complex=calc_lz_complexity(b,'primitive',0);
[q1,q2,q3]=calc_lz_complexity(b,'primitive',1);
[Cex,q2,q3]=calc_lz_complexity(b,'exhaustive',1);

T={'r', 'complex','my_complex';r,complex/length(b)*log2(length(b)),complex/length(b)};
disp(T)
T={'words','LZ-exhaustive','LZ-primitive'; complex,Cex,q1};
disp(T)

%------------------ LENGTH %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
trainLen= round(length(yy(:,1))/20)
testLen = round(length(yy(:,1))/1)
initLen = 1; 
data=yy(:,1);

% generate the ESN reservoir
inSize = 1; outSize = 1;
resSize = 2000;
a = 0.1; % leaking rate
rand( 'seed', 42 );
Win = (rand(resSize,1+inSize)-0.5) .* 1;

% dense W:
%W = rand(resSize,resSize)-0.5;

%sparse W:
%  W = sprand(resSize,resSize,0.01);
%  W_mask = (W~=0); 
%  W(W_mask) = (W(W_mask)-0.5);

 h2 = WattsStrogatz(resSize,4,0.4,3); % Creates WS network topology
 A1 = adjacency(h2); % retuns the sparce adjacency matrix of graph
 W = full(A1); % returns the full matrix

% normalizing and setting spectral radius
disp 'Computing spectral radius...';
opt.disp = 0;
rhoW = abs(eigs(W,1,'LM',opt));
disp 'done.'
W = W .* ( 1.25 /rhoW);

% allocated memory for the design (collected states) matrix
X = zeros(1+inSize+resSize,trainLen-initLen);

% set the corresponding target matrix directly
Yt = data(initLen+2:trainLen+1)';

% run the reservoir with the data and collect X
x = zeros(resSize,1);
for t = 1:trainLen
	u = data(t);
	x = (1-a)*x + a*tanh( Win*[1;u] + W*x );
	if t > initLen
		X(:,t-initLen) = [1;u;x];
	end
end

% train the output by ridge regression
reg = 1e-8;  % regularization coefficient
% direct equations from texts:
%X_T = X'; 
%Wout = Yt*X_T * inv(X*X_T + reg*eye(1+inSize+resSize));
% using Matlab solver:
Wout = ((X*X' + reg*eye(1+inSize+resSize)) \ (X*Yt'))'; 

% run the trained ESN in a generative mode. no need to initialize here, 
% because x is initialized with training data and we continue from there.
Y = zeros(outSize,testLen);
u = data(trainLen+1);
for t = 1:testLen 
	x = (1-a)*x + a*tanh( Win*[1;u] + W*x );
	y = Wout*[1;u;x];
	Y(:,t) = y;
	% generative mode:
	u = y;
	% this would be a predictive mode:
	%u = data(trainLen+t+1);
end

errorLen = 500;
mse = sum((data(trainLen+2:trainLen+errorLen+1)'-Y(1,1:errorLen)).^2)./errorLen;
disp( ['MSE = ', num2str( mse )] );

% plot some signals
figure(2);
clf
%plot( data(trainLen+2:trainLen+testLen+1), 'color', [0.6,0.6,.6],'LineWidth',1);
hold on;
plot( data, 'color', [0.6,0.6,.6],'LineWidth',1);
hold on;
plot( Y,'color', [0.7,0.0,.4],'LineWidth',1);
hold on
[pks,locs]=findpeaks(Y,'MinPeakHeight',0);
plot(locs,pks,'.','MarkerSize',10,'Color', [0.9 0.0 0.0])

[pks1,locs1]=findpeaks(-Y,'MinPeakHeight',0);
plot(locs1,-pks1,'.','MarkerSize',10,'Color', [0.1 0.0 0.9])
hold on
%str=zeros(length(locs)+length(locs1));
%length(str)

k=0;
for i=1:length(Y)
   for ii=1:length(locs) 
       if i==locs(ii)
           k=k+1;
%            T={'k','index';k,locs(ii)}
%            disp(T)
           str(k)=1;
       end
   end
       for iii=1:length(locs1) 
       if i==locs1(iii)
           k=k+1;
%           T={'k','index';k,locs1(iii)};
%            disp(T)
           str(k)=0;
       end
       end
end
length(str)
q = logical(str);
q2=q+1;
lookup_string1 = '10';
q1 = lookup_string1(q + 1)


[TRANS_EST, EMIS_EST] = hmmestimate(q2,q2);
A=TRANS_EST';
An=[A(2,2) A(2,1); A(1,2) A(1,1)]

%---------------------------------------------- !!!! 
fid1=fopen(name_seq_fake,'w');
for i=1:length(q)
      fprintf(fid1,'%d\n',q(i));
end
st=fclose (fid1);

complex=calc_lz_complexity(q,'primitive',0);
[q1,q2,q3]=calc_lz_complexity(q,'primitive',1);
[Cex,q2,q3]=calc_lz_complexity(q,'exhaustive',1);

T={'r', 'complex','my_complex';r,complex/length(b)*log2(length(b)),complex/length(b)};
disp(T)
T={'words','LZ-exhaustive','LZ-primitive'; complex,Cex,q1};
disp(T)


% xx=1:length(Y);
% TF = islocalmax(Y);
% TF1 = islocalmin(Y);
% plot(xx(TF),Y(TF),'.','MarkerSize',15,'Color', [0.1 0.1 0.1])
% hold on
% plot(xx(TF1),Y(TF1),'.','MarkerSize',15,'Color', [0.1 0.1 0.1])
% hold on

axis tight;
ylim([-55 55])
legend('Lorenz x-variable', 'Simulated x-variable');


end


function [value,isterminal,direction] = events(t,y);
if y(1)>0
    direction= [1];
else
    direction= [-1];
end
value= 10*(y(1)-y(2));
% this means x'=0 as x'=-10(x-y) in the Lorenz model 
isterminal=[0];
end 


