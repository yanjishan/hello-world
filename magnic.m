%matlab图像拼接(四种方法)
% 1、直接拼接，
% 2、亮度调整后拼接，
% 3、按距离比例融合，
% 4、亮度调整后按距离比例融合
 
%流程：
%1。读入左，右图，并取出重合部分，并转化为亮度图
%2。分别把每点的亮度值相加，得到一个比值
%3。把比值 乘以 右图
%4。再把左 各 右图 拼接
 
clear;close all,clc;
%读入原图 （左 右）
img1=imread('data\1.jpg');
img2=imread('data\2.jpg');
 figure;imshow(img1);%显示
%  figure;imshow(img2);
 %%
%查找它们的SIFT特征,并返回匹配点对---------------------特征匹配 开始
[des1, des2] = siftMatch(img1, img2);
des1=[des1(:,2),des1(:,1)];%左右（x和y）交换 为基础矩阵F 过滤匹配准备参数
des2=[des2(:,2),des2(:,1)];%
 
%用 基础矩阵F 过滤匹配的特征点对
matchs = matchFSelect(des1, des2) %匹配位置索引（掩码）
des1=des1(matchs,:);%取出内点
des2=des2(matchs,:);
 
% 画出匹配特征点的连接线（好点）
drawLinedCorner(img1,des1,img2, des2) ;
%------------------------------------------------------特征匹配 结束
 
[H,W,k]=size(img1);%图像大小
l_r=W-des1(1,2)+des2(1,2);%只取水平方向（第一个匹配点）重叠宽度
 %%
 
% 1、直接拼接-------------------------------------------------
 
%[H,W,k]=size(img1);
%l_r=405;%重叠宽度（W-宽 至 W）---如果不用特征匹配这里直接写重合区宽
L=W+1-l_r;%左边起点
R=W;%右边尾点
n=R-L+1;%重叠宽度：就是l_r
%直接拼接图
im=[img1,img2(:,n:W,:)];%1全图+2的后面部分
figure;imshow(im);title('直接拼接图');
 %%
% 2、亮度调整后拼接-------------------------------------------------
%根据之前H矩阵找到的两幅图的重叠（l_r）部分
A=img1(:,L:R,:);
B=img2(:,1:n,:);
%A、B 是对应在两幅图中的重叠区域
 
% A=uint8(A); figure;imshow(A);
% B=uint8(B);figure;imshow(B);
% 
[ma,na,ka]=size(A);
I1=rgb2gray(A);%转换为灰度图像
I1=double(I1);%转换为双精度
v1=0;
I2= rgb2gray(B);
I2=double(I2);
v2=0;
for i=1:ma
    for j=1:na
        %I1(i,j)=0.59*A(i,j,1)+0.11*A(i,j,2)+0.3*A(i,j,3);%按点转化为灰度图
        v1=v1+I1(i,j);%所有亮度值相加（和）
        %I2(i,j)=0.59*B(i,j,1)+0.11*B(i,j,2)+0.3*B(i,j,3);
        v2=v2+I2(i,j);
    end
end
 
%figure;imshow(I1,[]);
%figure;imshow(I2,[]);
 
%亮度比例，并按比例调整第二个图
k=v1/v2;
 
BB2=img2(:,n:W,:)*k;%乘比值
im2=[img1,BB2];%拼接
figure;imshow(im2);title('调整亮度后拼接图');
 
%% 
% 3、按距离比例融合-------------------------------------------------
% 4、亮度调整后按距离比例融合----------------------------------------
 
% 图像融合消除拼接缝隙
 
%用的渐入渐出融合即：距离权重融合
 
%[H,Y,t]=size(im);
C=im;%继承前图
D=im2;%继承前图（亮度）
% n=拼缝宽;
%for i=1:H %少一重循环
    for j=1:n
        d=1-(j)/n;%disp(d);% 距离权重
        C(1:H,L+j,:)=d*A(1:H,j,:)+(1-d)*B(1:H,j,:);%互补融合
        D(1:H,L+j,:)=d*A(1:H,j,:)+(1-d)*B(1:H,j,:)*k;
    end
%end
C=uint8(C);
figure;imshow(C);title('直接融合拼接图');%3
D=uint8(D);
figure;imshow(D);title('亮度处理后融合拼接图');%4
%%
% 此函数读取两幅图像，查找它们的SIFT特征
% 仅当匹配的距离小于与第二个最接近匹配的距离的阈值时，才接受匹配。
% 它返回两个图像的匹配点，matchLoc1 = [x1,y1;x2,y2;...]
%
% 

function [matchLoc1 matchLoc2] = siftMatch(img1, img2)

% 在每个图像查找 SIFT 特征点
[des1, loc1] = sift(img1);
[des2, loc2] = sift(img2);

% 对于MATLAB中的效率,计算单位向量之间的点乘比欧式距离更快捷。请注意: 
% 注意角的比率(单位矢量点积的反余弦)是小角度的欧氏距离之比的近似值。
%
% distRatio: 在这两队匹配中只保留矢量角从最近的第二近邻的比值小于distRatio的。
distRatio = 0.7;   

% 在第一个图像中的每个描述符，选择它的匹配到第二个图像。
des2t = des2';                          % 预计算矩阵转置 
matchTable = zeros(1,size(des1,1));
for i = 1 : size(des1,1)
   dotprods = des1(i,:) * des2t;        % 计算点积向量
   [vals,indx] = sort(acos(dotprods));  % 取逆余弦和排序结果

   % 看看和最近的邻居角比率小于distRatio。
   if (vals(1) < distRatio * vals(2))
      matchTable(i) = indx(1);
   else
      matchTable(i) = 0;
   end
end
% 保存匹配数据表

num = sum(matchTable > 0);
fprintf('找到 %d 对匹配点.\n', num);

idx1 = find(matchTable);
idx2 = matchTable(idx1);
x1 = loc1(idx1,2);
x2 = loc2(idx2,2);
y1 = loc1(idx1,1);
y2 = loc2(idx2,1);

matchLoc1 = [y1,x1];%把y坐标放前面
matchLoc2 = [y2,x2];

end