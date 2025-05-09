%% compute RMSE(MAE)|����RMSE(MAE) 
clear;close all;clc
% 1`modify the following directories 2`run|�޸�·��,������

% GT mask directory|��Ĥ·��
maskdir = '/Users/Hibbert/Desktop/srd3/results/';
MD = dir([maskdir '/*_inf.png']);

% maskdir = '/Users/Hibbert/Desktop/srd3/results';
%MD = dir([maskdir '/*_inf.png']);

% result directory|���·��
shadowdir = '//Users/Hibbert/Desktop/srd_sh2/results/';  
%%shadowdir = '/Users/Hibbert/Desktop/SD-results/ISTD+/results/'; 
%SD = dir([shadowdir '/*.png']);
SD = dir([shadowdir '/*_sr.png']);

% ground truth directory|GT·��
freedir = '/Users/Hibbert/Desktop/srd_sh2/results/'; %AISTD
FD = dir([freedir '/*_hr.png']);
total_dists = 0;
total_pixels = 0;
total_distn = 0;
total_pixeln = 0;
allmae=zeros(1,size(SD,1)); 
smae=zeros(1,size(SD,1)); 
nmae=zeros(1,size(SD,1)); 
ppsnr=zeros(1,size(SD,1));
ppsnrs=zeros(1,size(SD,1));
ppsnrn=zeros(1,size(SD,1));
sssim=zeros(1,size(SD,1));
sssims=zeros(1,size(SD,1));
sssimn=zeros(1,size(SD,1));
rrmse=zeros(1,size(SD,1));
rrmses=zeros(1,size(SD,1));
rrmsen=zeros(1,size(SD,1));
cform = makecform('srgb2lab');

for i=1:size(SD)
    %disp(SD(i));
    %disp(FD(i));
    %disp(MD(i));
    sname = strcat(shadowdir,SD(i).name); 
    fname = strcat(freedir,FD(i).name); 
    mname = strcat(maskdir,MD(i).name); 
    s=imread(sname);
    f=imread(fname);
    m=imread(mname);
    
    f = double(f)/255;
    s = double(s)/255;
    
    s=imresize(s,[256 256]);
    f=imresize(f,[256 256]);
    m = imresize(m, [256, 256]); % 调整尺寸到 256x256x3
    m = rgb2gray(m);             % 转换为 256x256

    



    nmask=~m;       %mask of non-shadow region|非阴影区域的mask
    smask=~nmask;   %mask of shadow regions|阴影区域的mask
    
    ppsnr(i)=psnr(s,f);

    % disp(size(s));      % 查看s的尺寸
    % disp(size(f));      % 查看f的尺寸
    % disp(size(smask));  % 查看smask的尺寸

    ppsnrs(i)=psnr(s.*repmat(smask,[1 1 3]),f.*repmat(smask,[1 1 3]));
    ppsnrn(i)=psnr(s.*repmat(nmask,[1 1 3]),f.*repmat(nmask,[1 1 3]));
    sssim(i)=ssim(s,f);
    sssims(i)=ssim(s.*repmat(smask,[1 1 3]),f.*repmat(smask,[1 1 3]));
    sssimn(i)=ssim(s.*repmat(nmask,[1 1 3]),f.*repmat(nmask,[1 1 3]));

   
    %% RMSE
     % arrmse=rmse(s,f);
     % rrmses(i)=rmse(s.*repmat(smask,[1 1 3]),f.*repmat(smask,[1 1 3]));
     % rrmsen(i)=rmse(s.*repmat(nmask,[1 1 3]),f.*repmat(nmask,[1 1 3]));

     %% MAE 作为 RMSE 替代
    dist = abs(s - f);  % 或者直接用 abs((f - s)) 结果一样
    smae_map = dist .* repmat(smask, [1 1 3]);
    nmae_map = dist .* repmat(nmask, [1 1 3]);
    
    rrmses(i) = sum(smae_map(:)) / sum(smask(:));  % RMSE(shadow)
    rrmsen(i) = sum(nmae_map(:)) / sum(nmask(:));  % RMSE(non-shadow)



    f = applycform(f,cform);    
    s = applycform(s,cform);
    
    %% MAE, per image
    dist=abs((f - s));
    sdist=dist.*repmat(smask,[1 1 3]);
    sumsdist=sum(sdist(:));
    ndist=dist.*repmat(nmask,[1 1 3]);
    sumndist=sum(ndist(:));
    
    sumsmask=sum(smask(:));
    sumnmask=sum(nmask(:));
    
    %% MAE, per pixel
    allmae(i)=sum(dist(:))/size(f,1)/size(f,2);
    smae(i)=sumsdist/sumsmask;
    nmae(i)=sumndist/sumnmask;
    
    total_dists = total_dists + sumsdist;
    total_pixels = total_pixels + sumsmask;
    
    total_distn = total_distn + sumndist;
    total_pixeln = total_pixeln + sumnmask;  


    disp(i);
end
fprintf('PSNR(all,non-shadow,shadow):\n%f\t%f\t%f\n',mean(ppsnr),mean(ppsnrn),mean(ppsnrs));
fprintf('SSIM(all,non-shadow,shadow):\n%f\t%f\t%f\n',mean(sssim),mean(sssimn),mean(sssims));
fprintf('RMSE(all,non-shadow,shadow):\n%f\t%f\t%f\n',mean(rrmse),mean(rrmsen),mean(rrmses));
fprintf('PI-Lab(all,non-shadow,shadow):\n%f\t%f\t%f\n',mean(allmae),mean(nmae),mean(smae));
fprintf('PP-Lab(all,non-shadow,shadow):\n%f\t%f\t%f\n\n',mean(allmae),total_distn/total_pixeln,total_dists/total_pixels); 