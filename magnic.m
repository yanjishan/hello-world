%matlabͼ��ƴ��(���ַ���)
% 1��ֱ��ƴ�ӣ�
% 2�����ȵ�����ƴ�ӣ�
% 3������������ںϣ�
% 4�����ȵ����󰴾�������ں�
 
%���̣�
%1����������ͼ����ȡ���غϲ��֣���ת��Ϊ����ͼ
%2���ֱ��ÿ�������ֵ��ӣ��õ�һ����ֵ
%3���ѱ�ֵ ���� ��ͼ
%4���ٰ��� �� ��ͼ ƴ��
 
clear;close all,clc;
%����ԭͼ ���� �ң�
img1=imread('data\1.jpg');
img2=imread('data\2.jpg');
 figure;imshow(img1);%��ʾ
%  figure;imshow(img2);
 %%
%�������ǵ�SIFT����,������ƥ����---------------------����ƥ�� ��ʼ
[des1, des2] = siftMatch(img1, img2);
des1=[des1(:,2),des1(:,1)];%���ң�x��y������ Ϊ��������F ����ƥ��׼������
des2=[des2(:,2),des2(:,1)];%
 
%�� ��������F ����ƥ����������
matchs = matchFSelect(des1, des2) %ƥ��λ�����������룩
des1=des1(matchs,:);%ȡ���ڵ�
des2=des2(matchs,:);
 
% ����ƥ��������������ߣ��õ㣩
drawLinedCorner(img1,des1,img2, des2) ;
%------------------------------------------------------����ƥ�� ����
 
[H,W,k]=size(img1);%ͼ���С
l_r=W-des1(1,2)+des2(1,2);%ֻȡˮƽ���򣨵�һ��ƥ��㣩�ص����
 %%
 
% 1��ֱ��ƴ��-------------------------------------------------
 
%[H,W,k]=size(img1);
%l_r=405;%�ص���ȣ�W-�� �� W��---�����������ƥ������ֱ��д�غ�����
L=W+1-l_r;%������
R=W;%�ұ�β��
n=R-L+1;%�ص���ȣ�����l_r
%ֱ��ƴ��ͼ
im=[img1,img2(:,n:W,:)];%1ȫͼ+2�ĺ��沿��
figure;imshow(im);title('ֱ��ƴ��ͼ');
 %%
% 2�����ȵ�����ƴ��-------------------------------------------------
%����֮ǰH�����ҵ�������ͼ���ص���l_r������
A=img1(:,L:R,:);
B=img2(:,1:n,:);
%A��B �Ƕ�Ӧ������ͼ�е��ص�����
 
% A=uint8(A); figure;imshow(A);
% B=uint8(B);figure;imshow(B);
% 
[ma,na,ka]=size(A);
I1=rgb2gray(A);%ת��Ϊ�Ҷ�ͼ��
I1=double(I1);%ת��Ϊ˫����
v1=0;
I2= rgb2gray(B);
I2=double(I2);
v2=0;
for i=1:ma
    for j=1:na
        %I1(i,j)=0.59*A(i,j,1)+0.11*A(i,j,2)+0.3*A(i,j,3);%����ת��Ϊ�Ҷ�ͼ
        v1=v1+I1(i,j);%��������ֵ��ӣ��ͣ�
        %I2(i,j)=0.59*B(i,j,1)+0.11*B(i,j,2)+0.3*B(i,j,3);
        v2=v2+I2(i,j);
    end
end
 
%figure;imshow(I1,[]);
%figure;imshow(I2,[]);
 
%���ȱ������������������ڶ���ͼ
k=v1/v2;
 
BB2=img2(:,n:W,:)*k;%�˱�ֵ
im2=[img1,BB2];%ƴ��
figure;imshow(im2);title('�������Ⱥ�ƴ��ͼ');
 
%% 
% 3������������ں�-------------------------------------------------
% 4�����ȵ����󰴾�������ں�----------------------------------------
 
% ͼ���ں�����ƴ�ӷ�϶
 
%�õĽ��뽥���ںϼ�������Ȩ���ں�
 
%[H,Y,t]=size(im);
C=im;%�̳�ǰͼ
D=im2;%�̳�ǰͼ�����ȣ�
% n=ƴ���;
%for i=1:H %��һ��ѭ��
    for j=1:n
        d=1-(j)/n;%disp(d);% ����Ȩ��
        C(1:H,L+j,:)=d*A(1:H,j,:)+(1-d)*B(1:H,j,:);%�����ں�
        D(1:H,L+j,:)=d*A(1:H,j,:)+(1-d)*B(1:H,j,:)*k;
    end
%end
C=uint8(C);
figure;imshow(C);title('ֱ���ں�ƴ��ͼ');%3
D=uint8(D);
figure;imshow(D);title('���ȴ�����ں�ƴ��ͼ');%4
%%
% �˺�����ȡ����ͼ�񣬲������ǵ�SIFT����
% ����ƥ��ľ���С����ڶ�����ӽ�ƥ��ľ������ֵʱ���Ž���ƥ�䡣
% ����������ͼ���ƥ��㣬matchLoc1 = [x1,y1;x2,y2;...]
%
% 

function [matchLoc1 matchLoc2] = siftMatch(img1, img2)

% ��ÿ��ͼ����� SIFT ������
[des1, loc1] = sift(img1);
[des2, loc2] = sift(img2);

% ����MATLAB�е�Ч��,���㵥λ����֮��ĵ�˱�ŷʽ�������ݡ���ע��: 
% ע��ǵı���(��λʸ������ķ�����)��С�Ƕȵ�ŷ�Ͼ���֮�ȵĽ���ֵ��
%
% distRatio: ��������ƥ����ֻ����ʸ���Ǵ�����ĵڶ����ڵı�ֵС��distRatio�ġ�
distRatio = 0.7;   

% �ڵ�һ��ͼ���е�ÿ����������ѡ������ƥ�䵽�ڶ���ͼ��
des2t = des2';                          % Ԥ�������ת�� 
matchTable = zeros(1,size(des1,1));
for i = 1 : size(des1,1)
   dotprods = des1(i,:) * des2t;        % ����������
   [vals,indx] = sort(acos(dotprods));  % ȡ�����Һ�������

   % ������������ھӽǱ���С��distRatio��
   if (vals(1) < distRatio * vals(2))
      matchTable(i) = indx(1);
   else
      matchTable(i) = 0;
   end
end
% ����ƥ�����ݱ�

num = sum(matchTable > 0);
fprintf('�ҵ� %d ��ƥ���.\n', num);

idx1 = find(matchTable);
idx2 = matchTable(idx1);
x1 = loc1(idx1,2);
x2 = loc2(idx2,2);
y1 = loc1(idx1,1);
y2 = loc2(idx2,1);

matchLoc1 = [y1,x1];%��y�����ǰ��
matchLoc2 = [y2,x2];

end