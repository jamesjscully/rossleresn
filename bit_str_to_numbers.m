function bit_str_to_numbers(bit_to_count)

% bit_to_count =  number of bits to be considered

load('lor_par_data.mat')

[sigma,r] = meshgrid(sigma_,r_);

symbol_vector = '01';

Ni = size(bit_string,1);
Nj = size(bit_string,2);

sumMatrix = zeros(Ni,Nj);


for i=1:Ni
    parfor j=1:Nj
       sumMatrix(i,j) =  bin2dec(symbol_vector(bit_string{i,j}(bit_to_count:-1:1)+1));
    end
    
    h = waitbar(i/Ni);
    
end

close(h)

figure
mesh(r',sigma',sumMatrix)
view(0,90)
xlabel('\sigma')
ylabel('r')
title(['N bits = ',num2str(bit_to_count)])

end