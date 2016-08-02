%% sample ER graph & look at it

n=50;
p=0.5;
ER=rand(n)>1-p;
spy(ER)

%% sample large ER graph & look at it


n=10^5;
p=10^-3;
m=binornd(n^2, p);
pi=randperm(n^2,m);
A=zeros(n);
A(pi)=1;
spy(A)

%% plot same graph but randomly resorted

pi=randperm(n);
spy(A(pi,pi));

%% sort by degree

n=250;
p=0.1;
A=rand(n)>1-p;
spy(A)
deg=sum(A);
[B,I]=sort(deg);
spy(A(I,I))
hist(deg)

%% estimate p

n=250;
p=0.1;
A=rand(n)>1-p;

phat=sum(A(:))/n^2

%% compute error vs n

p=0.01;
nvec=1:4:200;
phat=zeros(1,length(nvec));
i=0;
for n=nvec
    i=i+1;
    A=rand(n)>1-p;
    phat(i)=sum(A(:))/(n^2);
end
plot(nvec,(phat-p).^2,'.-','markersize',12)
set(gca,'Xscale','log')

%% for loops for sampling indenpendent edge random graph

clear
n=100;
P=0.1*rand(n);
for i=1:n
    for j=1:n
        A(i,j)=rand>1-P(i,j);
    end
end
spy(A)

%% sort IERG by degree

n=50;
P=0.2*rand(n);

A=rand(n)>1-P;
spy(A)
deg=sum(A);
DEG=sum(P);
[B,I]=sort(DEG);

subplot(131), imagesc(P(I,I)), axis('square')
subplot(132)
imagesc(A(I,I)), axis('square')

subplot(133)
hist(deg)

%% error as a function of sparsity

clc, clf
n=10;
cvec=[0.1, 0.9];
for c=1:2
    k=cvec(c);
    P=rand(n)*k;
    m=100;
    
    A=zeros(n,n,m);
    for i=1:m
        A(:,:,i)=rand(n)>1-P;
        Phat=sum(A,3)/m;
        err(i)=sum(sum((Phat-P).^2))
    end
    
    Phat=sum(A,3);
    
    subplot(131)
    imagesc(P), axis('square')
    
    subplot(132)
    imagesc(Phat), axis('square')
    
    subplot(133), hold all
    plot(1:m,err)
end
set(gca,'Yscale','log')
legend('0.1','0.9')


%% error as a function of number of vertices

clear
clc, clf
c=0.1;
m=50;
nvec=10:10:200;
maxn=max(nvec);

for ns=1:length(nvec)
    n=nvec(ns);
    P=c*rand(n);
    A=zeros(n,n,m);
    for i=1:m
        A(:,:,i)=rand(n)>1-P;
    end
    Phat=sum(A,3)/m;
    err(ns)=sum(sum((Phat-P).^2));
end

subplot(131), imagesc(P), axis('square'), title('P')
subplot(132), imagesc(Phat), axis('square'), title('Phat')
subplot(133), plot(nvec,err), 
hold all, plot(nvec,err./nvec.^2,'r')
ylabel('err'), xlabel('number of vertices')
set(gca,'Yscale','log')
legend('err','err/n^2','location','best')

%% sample and plot stochastic block model

clear, clf
n=100;
B=[0.5, 0.2; 0.2, 0.5];
pi=0.5;
Z=(rand(n,1)>pi)+1;

A=zeros(n);
for i=1:n
    for j=1:n
        A(i,j)=rand<B(Z(i),Z(j));
    end
end
[~,I]=sort(Z);

subplot(121), spy(A)
subplot(122), spy(A(I,I))

%% estimate B & pi

clc
pihat=sum(Z-1)/n

n1=length(find(Z==1));

for i=1:2
    for j=1:2
Bhat(i,j)=sum(sum(A(Z==i,Z==j)))/n1^2;
    end
end

Bhat

pi
B


%% sample ER graphs and classes jointly


m=100;
n=70;

p0=0.2;
p1=0.1;

pi=0.5;

A=zeros(n,n,m);
Y=zeros(m,1);
for i=1:m
    Y(i)=rand>pi;
    if Y(i)==1
       A(:,:,i)=rand(n)<p1; 
    else
        A(:,:,i)=rand(n)<p0;
    end
end

i=1;
i=i+1; spy(A(:,:,i)), title(Y(i))

%% sample IERG graphs and classes jointly

m=100;
n=70;

p0=rand(n);
p1=p0*0.2;

pi=0.5;

A=zeros(n,n,m);
Y=zeros(m,1);
for i=1:m
    Y(i)=rand>pi;
    if Y(i)==1
       A(:,:,i)=rand(n)<p1; 
    else
        A(:,:,i)=rand(n)<p0;
    end
end


i=1;
i=i+1; spy(A(:,:,i)), title(Y(i))


%%

clear, clf
n=100;
B=[0.5, 0.2; 0.2, 0.5];
pi=0.5;
Z=(rand(n,1)>pi)+1;

A=zeros(n);
for i=1:n
    for j=1:n
        A(i,j)=rand<B(Z(i),Z(j));
    end
end
[z,I]=sort(Z);
A=A(I,I);


[V,D]=eigs(A,2);
idx=kmeans(V,2);

err1=sum((idx-z).^2);
err2=sum((idx-2./z).^2);

err=min(err1,err2)


%%
figure(1), clf
subplot(131), spy(A)
subplot(132)
plot(1:n,idx), hold all
plot(1:n,z)

subplot(133), hold all
plot(V(idx==1,:),'r.')
plot(V(idx==2,:),'b.')