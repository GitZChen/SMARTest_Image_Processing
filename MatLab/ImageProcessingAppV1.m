clc; clear all; close all;

% Read image
i = imread('PSF8_square.jpg');
% i = imread('Cropped_Multiplex_InstiTest.jpg');
% i = imread('HIV_Negative_Android_3162017.jpg');
% i = imread('HIV_Positive_0516.jpg');
% i = imread('HIV_Positive_0516_Residue.jpg');
% figure('units','normalized','outerposition',[0.1 0.1 0.7 0.6]);
figure(1)
subplot(2,2,1), imshow(i); title('1: Starting image', 'FontSize', 14);

% Get a grayscale image
ig = i(:,:,1); 
subplot(2,2,2), imshow(ig); title('2: Grayscale image', 'FontSize', 14);

% Circular mask generation
[imageSizeY, imageSizeX] = size(ig);
[columnsInImage, rowsInImage] = meshgrid(1:imageSizeX, 1:imageSizeY);
center = [imageSizeX/2, imageSizeY/2];

% HIV negative sample
% centerX = 2500;
% centerY = 1530;
% radius = 335;

% HIV positive sample
% centerX = 905;
% centerY = 1085;
% radius = 300;

% Multiplex Insti
% centerX = 300;
% centerY = 343;
% radius = 75; 

% HIV positive sample 05/16/2017
% centerX = 1539;
% centerY = 1244;
% radius = 250;

% HIV positive sample w/ residue 05/16/2017
% centerX = 1408;
% centerY = 1239;
% radius = 200;

%PSF9 square
centerX = 512;
centerY = 512;
radius = 160;

circlePixels = (rowsInImage - centerY).^2 + (columnsInImage - centerX).^2 <= radius.^2;
colormap([0 0 0; 1 1 1]);

subplot(2,2,3), imshow(circlePixels); title('3: Circular mask', 'FontSize', 14);

% Get area of interest using mask
maskedImage = ig;
maskedImage(~circlePixels) = 0;
subplot(2,2,4), imshow(maskedImage); title('4: Area of interest', 'FontSize', 14);

% Convert to a binary image
BWd = imbinarize(maskedImage, 'adaptive','ForegroundPolarity','dark','Sensitivity',0.5);
BWb = imbinarize(maskedImage, 'adaptive','ForegroundPolarity','bright','Sensitivity',0.5);

figure(2);
subplot(2,2,1), imshow(BWd); title('5: Binary image (Dark)', 'FontSize', 14);
subplot(2,2,2), imshow(BWb); title('5: Binary image (Bright)', 'FontSize', 14);

% Determine edges using Canny algorithm
[edgeimage_d, threshout_d] = edge(BWd, 'Canny');
[edgeimage_b, threshout_b] = edge(BWb, 'Canny');
subplot(2,2,3), imshow(edgeimage_d); title('6: Edges marked (Dark)', 'FontSize', 14);
subplot(2,2,4), imshow(edgeimage_b); title('6: Edges marked (Bright)', 'FontSize', 14);

% Use imfindcircles to find circles within the specified radius range
[centersDark, radiiDark] = imfindcircles(edgeimage_d, [20 40], 'ObjectPolarity', 'dark', 'Sensitivity', 0.9,'Method', 'TwoStage');
[centersBright, radiiBright] = imfindcircles(edgeimage_b, [20 40], 'ObjectPolarity', 'bright', 'Sensitivity', 0.85,'Method', 'TwoStage');

% [centersDark, radiiDark] = imfindcircles(ig, [20 40], 'ObjectPolarity', 'dark', 'Sensitivity', 0.9,'Method', 'TwoStage');
% [centersBright, radiiBright] = imfindcircles(ig, [20 40], 'ObjectPolarity', 'bright', 'Sensitivity', 0.85,'Method', 'TwoStage');

figure(3); 
imshow(i); title('Input image with zones marked', 'FontSize', 14);
hDark = viscircles(centersDark, radiiDark,'LineStyle','-', 'EdgeColor','r');
hBright = viscircles(centersBright, radiiBright, 'LineStyle','-', 'EdgeColor','b');
hold on
centers_1 = [reshape(centersDark',1,numel(centersDark)), reshape(centersBright',1,numel(centersBright))];
radii = [radiiDark', radiiBright'];

j = 1;
centers_x = zeros(1,length(centers_1)/2);
centers_y = zeros(1,length(centers_1)/2);
for i = 1:2:length(centers_1)
    centers_x(j) = centers_1(i);
    centers_y(j) = centers_1(i+1);
    j = j + 1;
end

centers = [centers_x;centers_y];

% Results: 1 = HIV positive, 0 = HIV negative, 2 = invalid result
if numel(centers)>= 4
    
    distances = euclideandist(centers, radii);
    condition = (distances > 110).*(distances < 150);
    
    if sum(condition) > 0
        result = 1;
    else
        result = 0;
    end
    
elseif numel(centers) < 4 && numel(centers) > 0
    
    result = 0;
    
elseif numel(centers) == 0
    
    result = 2;
end
display(result);
hold off;