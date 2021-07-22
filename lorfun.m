function by= lorfun(r,tmx)
options = odeset('Events',@events);
y0=[0.001,0.,0.];
tu=[0 100];
tmax=[0 tmx];

[t,y]=ode45(@(t,s)loreq(t,s,r),tu,y0);

y0=y(length(t),:);
[t,y,tau1,ye1,ie1]=ode45(@(t,s)loreq(t,s,r),tmax,y0,options);

by = ye1(:,1)';
end

function dydt = loreq(t,s,r)
dydt=zeros(3,1);
dydt(1)=-10*(s(1)-s(2));
dydt(2)=r*s(1)-s(2)-s(1).*s(3);
dydt(3)= -2.666666*s(3)+s(1).*s(2);
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
