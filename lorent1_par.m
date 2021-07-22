clear all
name='entr_par_n1.dat';
tmx=3000; 
Lmax=6; % max length of words
rr=28:0.25:30; nr=length(rr);
ent=zeros(Lmax,nr);

tic
for ir=1:nr
  r=rr(ir);
  by= lorfun(r,tmx);
  b = int8(heaviside(by));
  ent(:,ir)=entr_n(b,Lmax);
  
end
toc

fid1=fopen(name,'w');
for ir=1:nr
   fprintf(fid1,'%12.5g %12.5g %12.5g %12.5g %12.5g %12.5g %12.5g\n',rr(ir),ent(1,ir),ent(2,ir),ent(3,ir),ent(4,ir),ent(5,ir),ent(6,ir));
end
st=fclose (fid1);



