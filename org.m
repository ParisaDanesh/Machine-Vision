faceDetector = vision.CascadeObjectDetector;
%EyeDetect = vision.CascadeObjectDetector('EyePairSmall');
NoseDetect = vision.CascadeObjectDetector('Nose');
%MouthDetect = vision.CascadeObjectDetector('Mouth');
emotionfiles = dir('./Datasets/emotion/face em*');
Features = [];
for j=1:length(emotionfiles)
    direct = emotionfiles(j).name;
    imagefiles = dir(strcat('./Datasets/emotion/',direct,'/*.tiff'));
    for i=1:length(imagefiles)
        name = strcat('./Datasets/emotion/',direct,'/',imagefiles(i).name);
        I = imread(name);

        %-------------------------------detect face----------------------------
        bbox = step(faceDetector , I );
        cropImage = imcrop(I,bbox);
        
        %-----------------------------detect Nose------------------------------
        BB=step(NoseDetect,cropImage);
        cropNose = imcrop(cropImage ,[BB(1,1),BB(1,2)-10,BB(1,3),BB(1,4)]);
        imwrite(cropNose ,['./me/nose/',num2str(j),'/',num2str(i),'.jpg']);
            %blockbandi va miangine shedata
            cropNose = imresize(cropNose,[20 30]);
            cropNoseBlock = blockproc(cropNose,[4 5],@(x)mean(x.data(:)));
            cropNoseBlock = reshape(cropNoseBlock , 1 ,[]);
        %-------------------------------detect eyes----------------------------

            %detect leftEyes
            cropLeftEyes = imcrop(cropImage ,[BB(1,1)-33,BB(1,2)-29,BB(1,3)-2,BB(1,4)-7]);
            imwrite(cropLeftEyes ,['./me/leftEyes/',num2str(j),'/',num2str(i),'.jpg']);
                %edge
                cropLeftEyesEdge = edge(cropLeftEyes,'log');
                %block bandi va sheddatgiri
                cropLeftEyesEdge = imresize(cropLeftEyesEdge,[20 30]);
                cropLeftEyesEdgeBlock = blockproc(cropLeftEyesEdge,[4 5],@(x)mean(x.data(:)));
                cropLeftEyesEdgeBlock = reshape(cropLeftEyesEdgeBlock , 1 ,[]);
            %detect rightEyes
            cropRightEyes = imcrop(cropImage ,[BB(1,1)+33,BB(1,2)-30,BB(1,3)-2,BB(1,4)-7]);
            imwrite(cropRightEyes ,['./me/rightEyes/',num2str(j),'/',num2str(i),'.jpg']);
                %edge
                cropRightEyesEdge = edge(cropRightEyes,'log');
                %block bandi
                cropRightEyesEdge = imresize(cropRightEyesEdge,[20 30]);
                cropRightEyesEdgeBlock = blockproc(cropRightEyesEdge,[4 5],@(x)mean(x.data(:)));
                cropRightEyesEdgeBlock = reshape(cropRightEyesEdgeBlock,1,[]);
        %-------------------------------detect eyebrows------------------------
        cropEyebrows = imcrop(cropImage ,[BB(1,1)-40,BB(1,2)-51,BB(1,3)+75,BB(1,4)-15]);
        imwrite(cropEyebrows ,['./me/eyebrows/',num2str(j),'/',num2str(i),'.jpg']);
        cropEyebrows = imresize(cropEyebrows , [20 120]);
            %SIFT
            imageSingle = single(cropEyebrows);
            siftFeatures = vl_sift(imageSingle,'Levels',20);
            siftFeatures = siftFeatures(1:4,1:5);
            siftFeatures = reshape(siftFeatures,1,[]);

        %----------------------------detect mouth------------------------------
        cropMouth = imcrop(cropImage ,[BB(1,1)-5,BB(1,2)+30,BB(1,3)+12,BB(1,4)+15]);
        imwrite(cropMouth ,['./me/mouth/',num2str(j),'/',num2str(i),'.jpg']);
            %wavelet 
            cropMouth = imresize(cropMouth,[30 30]);
            [cA,cH,cV,cD]=dwt2(cropMouth,'haar');
            temp=blockproc(cH, [5 5], @(x) mean(x.data(:)));
            cH=reshape(temp,1,[]);
            temp=blockproc(cV, [5 5], @(x) mean(x.data(:)));
            cV=reshape(temp,1,[]);
            temp=blockproc(cD, [5 5], @(x) mean(x.data(:)));
            cD=reshape(temp,1,[]);
            X = horzcat(cH,cV,cD);
        %-----------------
        tempFeature = horzcat(cropNoseBlock , cropLeftEyesEdgeBlock , cropRightEyesEdgeBlock, siftFeatures,X,j); %shuffle , train 70 , test 30
        Features = vertcat(Features,tempFeature);
    end
end
%xlswirte('FeatureExtractions.xlsx',Features);
%-----------------shuffle
rows = size(Features,1);
cols = size(Features,2);
a = [];
Features = Features(randperm(rows),:);
for i=1 : 20
   a = [a ; randperm(cols -1 , 80)];
end

%-------pca test
% X1 = Features(:,:);
% [a,b,c] = pca(X1);
% X2 = X1*a(:,1:9);
% Features = [X2 , Features];

%------------------normalize features
normalizedFeatures = zeros(rows,cols -1);
trainmax = max(Features);
trainmin = min(Features);
for j =1 : rows
    for i=1 : cols -1

        normalizedFeatures(j,i) = (trainmax(1,i) - Features(j,i)) / (trainmax(1,i) - trainmin(1,i));

    end
end

normalizedFeatures = horzcat(normalizedFeatures,Features(:,cols) );
% X = normalizedFeatures(:,1:137);
% [a,b,c] = pca(X);
% X2 = X*a(:,1:9);
% normalizedFeatures = [X2 , normalizedFeatures(:,138)];
% 
% nrows = size(normalizedFeatures,1);
% ncols = size(normalizedFeatures,2);

%------------------train&test data  (70% train , 30% test )
trainrow = 70 * rows/100;
train = normalizedFeatures(1:trainrow , 1:cols-1);
trainClass = normalizedFeatures(1:trainrow , cols);
test = normalizedFeatures(trainrow+1 : rows , 1: cols -1);
testClass = normalizedFeatures(trainrow+1:rows , cols);

%------------------train 

Mdl = fitcecoc(train,trainClass);
Group = predict(Mdl,test);

%----------------recall o ... 
fp = zeros(1,7);
tp = zeros(1,7);
fn = zeros(1,7);
tn = zeros(1,7);
classes = [1,2,3,4,5,6,7];

for i =1 :7
   for j =1 : size(Group,1)
       if isequal(Group(j,:), classes(1,i)) && isequal(testClass(j,:), classes(1,i))
           tp(1,i) = tp(1,i) + 1; 
       elseif isequal(Group(j,:), classes(1,i)) && ~isequal(testClass(j,:), classes(1,i))
           fp(1,i) = fp(1,i) +1;
       elseif ~isequal(Group(j,:), classes(1,i)) && isequal(testClass(j,:), classes(1,i))
           fn(1,i) = fn(1,i) +1;
       elseif ~isequal(Group(j,:), classes(1,i)) && ~isequal(testClass(j,:), classes(1,i))
           tn(1,i) = fn(1,i) +1;
       end
   end
end

pre = tp ./ (tp + fp);
rec = tp ./ (tp + fn);

totalpre = sum(pre)/7;
totalrec = sum(rec)/7;

fmean = 2* (pre .* rec) ./ (pre + rec);

totfmean = sum(fmean)/7;

accur = (tp + tn) ./ ( tp + tn + fp + fn);
totaccur = sum(accur)/7;
comatrix = zeros(7,7);
for i= 1 : size(Group,1)
   for j=1 : 7
      if isequal(testClass(i,:) , classes(1,j))
          for k =1 : 7
             if isequal(Group(i,:) , classes(1,k))
                 comatrix(j,k) = comatrix(j,k) +1;
             end
          end
      end
   end
end