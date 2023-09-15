clear all;
clc;
file_path ='D:\plot\Test\2\';    %需要处理图片的路径
img_path_list = dir(strcat(file_path,'*.jpg'));      %获取该文件夹中所有jpg格式的图片   
img_num = length(img_path_list)  %获得图像的总数量
    for j=1:img_num  %逐一读取图片
        image_name = img_path_list(j).name;   %图片名
        N=image_name(1:end-4);
        image = imread(strcat(file_path,image_name));   
%         image=imadjust(image,[0.2 0.3 0;0.6 0.7 1],[]);

%   for k=1:3
% %image(:,:,k)=flipud(image(:,:,k));%上下翻转
% image(:,:,k)=fliplr(image(:,:,k));%左右翻转
%  end
%image=imrotate(image,50,'bilinear');%旋转
%image=histeq(image);%直方图
  image = imresize(image,[227,227]); %重构
 imwrite(image,strcat('D:\plot\Testnew\2\',N,'.jpg')); 
            
    end






    
