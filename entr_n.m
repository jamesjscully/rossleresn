function dh = entr_n(d,Lmax)
n=length(d);
n_letters = 2;
h=zeros(1,Lmax);
lmax=Lmax+1;
for j=1:lmax
  nw=2^j;
  cnt=zeros(1,nw);
  % generate dictionary
  C = cell(1, j);
  [C{:}] = ndgrid(0:n_letters - 1);
  C = reshape(cat(j+1, C{:}), [], j);
  C=int8(C);
  
  n1=n-mod(n,j);
  zk = reshape(d(1:n1), j, []);
  m=length(unique(zk','row'));
  correction=(m-1)/(2*n);
  for i=1:nw
     cnt(i)=length(strfind(d,C(i,:)));
  end
  h(j)=-correction;
  s=sum(cnt);
  if s>0
    for i=1:nw
      if cnt(i)>0
        p=cnt(i)/s;
        h(j)=h(j)-p*log2(p);
      end
    end
  end
end
dh=diff(h);
end

