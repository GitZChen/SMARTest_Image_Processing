function varargout = ImageProcessingGUI(varargin)
% IMAGEPROCESSINGGUI MATLAB code for ImageProcessingGUI.fig
%      IMAGEPROCESSINGGUI, by itself, creates a new IMAGEPROCESSINGGUI or raises the existing
%      singleton*.
%
%      H = IMAGEPROCESSINGGUI returns the handle to a new IMAGEPROCESSINGGUI or the handle to
%      the existing singleton*.
%
%      IMAGEPROCESSINGGUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in IMAGEPROCESSINGGUI.M with the given input arguments.
%
%      IMAGEPROCESSINGGUI('Property','Value',...) creates a new IMAGEPROCESSINGGUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before ImageProcessingGUI_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to ImageProcessingGUI_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help ImageProcessingGUI

% Last Modified by GUIDE v2.5 17-May-2017 00:48:01

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
    'gui_Singleton',  gui_Singleton, ...
    'gui_OpeningFcn', @ImageProcessingGUI_OpeningFcn, ...
    'gui_OutputFcn',  @ImageProcessingGUI_OutputFcn, ...
    'gui_LayoutFcn',  [] , ...
    'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before ImageProcessingGUI is made visible.
function ImageProcessingGUI_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to ImageProcessingGUI (see VARARGIN)
clc;
% Choose default command line output for ImageProcessingGUI
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes ImageProcessingGUI wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = ImageProcessingGUI_OutputFcn(hObject, eventdata, handles)
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in load_image.
function load_image_Callback(hObject, eventdata, handles)
% hObject    handle to load_image (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
axes(handles.axes1);
folder_name = uigetdir('G:\Google Drive\Course documents\Columbia Courses\PhD Research\PROJECTS\Mchip\Dongle stuff',...
    'Choose folder');
cd (folder_name);

image_list = dir('*.jpg');
number_files = length(image_list);
image_names = {image_list.name};
[s, ok] = listdlg('PromptString','Select the image file','ListSize',[300 300],'SelectionMode','single','ListString',image_names);

if ok == 0
    disp('No image selected');
    return;
end
i = imread(image_names{s});
handles.i = i;
handles.s = s;
handles.image_names = image_names;
guidata(hObject, handles);
imshow(i); title(image_names(s), 'interpreter','none');

% --- Executes on button press in distance_toggle.
function distance_toggle_Callback(hObject, eventdata, handles)
% hObject    handle to distance_toggle (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of distance_toggle
axes(handles.axes1);
button_state = get(hObject,'Value');

if button_state == get(hObject,'Min')
    dist_handle = handles.dist_handle;
    api = iptgetapi(dist_handle);
    api.delete();
elseif button_state == get(hObject, 'Max')
    dist_handle = imdistline(gca);
    api = iptgetapi(dist_handle);
%     setLabelVisible(dist_handle,false);
end
handles.dist_handle = dist_handle;
guidata(hObject, handles);


% --- Executes on button press in export_distance.
function export_distance_Callback(hObject, eventdata, handles)
% hObject    handle to export_distance (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
dist_handle = handles.dist_handle;

api = iptgetapi(dist_handle);
radius = api.getDistance()
pos = api.getPosition();
centerX = pos(1)
centerY = pos(3)

handles.centerX = centerX;
handles.centerY = centerY;
handles.radius = radius;
api.delete();
guidata(hObject, handles);

% --- Executes on button press in pre_process_image.
function pre_process_image_Callback(hObject, eventdata, handles)
% hObject    handle to pre_process_image (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
axes(handles.axes1);
i = handles.i;
centerX = handles.centerX;
centerY = handles.centerY;
radius = handles.radius;

ig = i(:,:,1);

[imageSizeY, imageSizeX] = size(ig);
[columnsInImage, rowsInImage] = meshgrid(1:imageSizeX, 1:imageSizeY);
center = [imageSizeX/2, imageSizeY/2];

% [points_x, points_y] = ginput(2) 
% centerX = points_x(1);
% centerY = points_y(1);
% radius = sqrt((points_x(1) - points_x(2))^2 + (points_y(1) - points_y(2))^2);

% HIV negative sample
% centerX = 2500;
% centerY = 1530;
% radius = 335;

% HIV positive sample
% centerX = 905;
% centerY = 1085;
% radius = 300;

% HIV positive sample 05/16/2017
% centerX = 1539;
% centerY = 1244;
% radius = 200;

% HIV positive sample w/ residue 05/16/2017
% centerX = 1408;
% centerY = 1239;
% radius = 144;

% Multiplex Insti
% centerX = 300;
% centerY = 343;
% radius = 75;

circlePixels = (rowsInImage - centerY).^2 + (columnsInImage - centerX).^2 <= radius.^2;
maskedImage = ig;
maskedImage(~circlePixels) = 0;

BWd = imbinarize(maskedImage, 'adaptive','ForegroundPolarity','dark','Sensitivity',0.5);
BWb = imbinarize(maskedImage, 'adaptive','ForegroundPolarity','bright','Sensitivity',0.5);
[edgeimage_d, threshout_d] = edge(BWd, 'Canny');
[edgeimage_b, threshout_b] = edge(BWb, 'Canny');

handles.edgeimage_d = edgeimage_d;
handles.edgeimage_b = edgeimage_b;
guidata(hObject, handles);

% --- Executes on button press in process_both.
function process_both_Callback(hObject, eventdata, handles)
% hObject    handle to process_both (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
axes(handles.axes1);
i = handles.i;
s = handles.s;
image_names = handles.image_names;
edgeimage_d = handles.edgeimage_d;
edgeimage_b = handles.edgeimage_b;

[centersDark, radiiDark] = imfindcircles(edgeimage_d, [20 50], 'ObjectPolarity', 'dark', 'Sensitivity', 0.9,'Method', 'TwoStage');
[centersBright, radiiBright] = imfindcircles(edgeimage_b, [20 50], 'ObjectPolarity', 'dark', 'Sensitivity', 0.85,'Method', 'TwoStage');
% [centersDark, radiiDark] = imfindcircles(edgeimage_d, [10 20], 'ObjectPolarity', 'dark', 'Sensitivity', 0.9,'Method', 'TwoStage');
% [centersBright, radiiBright] = imfindcircles(edgeimage_b, [10 20], 'ObjectPolarity', 'dark', 'Sensitivity', 0.85,'Method', 'TwoStage');

imshow(i); title(image_names(s), 'interpreter','none');
hold on;
hDark = viscircles(centersDark, radiiDark,'LineStyle','-', 'EdgeColor','r');
hBright = viscircles(centersBright, radiiBright, 'LineStyle','-', 'EdgeColor','b');
hold off;
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

if  numel(centers)>= 4
    distances = euclideandist(centers, radii);
else
    distances = 0;
end

handles.distances_value = distances;
handles.radii = radii;
handles.centers = centers;
guidata(hObject, handles);

% --- Executes on button press in bright_foreground_process.
function bright_foreground_process_Callback(hObject, eventdata, handles)
% hObject    handle to bright_foreground_process (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
axes(handles.axes1);
i = handles.i;
s = handles.s;
image_names = handles.image_names;
edgeimage_b = handles.edgeimage_b;

% [centersBright, radiiBright] = imfindcircles(edgeimage_b, [10 20], 'ObjectPolarity', 'dark', 'Sensitivity', 0.85,'Method', 'TwoStage');
[centersBright, radiiBright] = imfindcircles(edgeimage_b, [20 50], 'ObjectPolarity', 'dark', 'Sensitivity', 0.85,'Method', 'TwoStage');
centersDark = [];
radiiDark = [];

imshow(i); title(image_names(s), 'interpreter','none');
hold on;
hBright = viscircles(centersBright, radiiBright, 'LineStyle','-', 'EdgeColor','b');
hold off;
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

if numel(centers)>= 4
    distances = euclideandist(centers, radii);
else
    distances = 0;
end

handles.distances_value = distances;
handles.radii = radii;
handles.centers = centers;
guidata(hObject, handles);


% --- Executes on button press in dark_foreground_process.
function dark_foreground_process_Callback(hObject, eventdata, handles)
% hObject    handle to dark_foreground_process (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
axes(handles.axes1);
i = handles.i;
s = handles.s;
image_names = handles.image_names;
edgeimage_d = handles.edgeimage_d;

% [centersDark, radiiDark] = imfindcircles(edgeimage_d, [10 20], 'ObjectPolarity', 'dark', 'Sensitivity', 0.9,'Method', 'TwoStage');
[centersDark, radiiDark] = imfindcircles(edgeimage_d, [20 50], 'ObjectPolarity', 'dark', 'Sensitivity', 0.9,'Method', 'TwoStage');
centersBright = [];
radiiBright = [];

imshow(i); title(image_names(s), 'interpreter','none');
hold on;
hDark = viscircles(centersDark, radiiDark,'LineStyle','-', 'EdgeColor','r');
hold off;
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

if  numel(centers)>= 4
    distances = euclideandist(centers, radii);
else
    distances = 0;
end

handles.distances_value = distances;
handles.radii = radii;
handles.centers = centers;
guidata(hObject, handles);

% --- Executes on button press in result.
function result_Callback(hObject, eventdata, handles)
% hObject    handle to result (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
radii = handles.radii;
centers = handles.centers;
distances = handles.distances_value;

if numel(centers) < 4
    
    set(handles.result_text, 'String', 'HIV Negative');
    
elseif numel(centers) >= 4
    
    condition = (distances > 150).*(distances < 200);
    
    if sum(condition) > 0
        set(handles.result_text, 'String', 'HIV Positive');
    else
        set(handles.result_text, 'String', 'HIV Negative');
    end
end
set(handles.distances_text, 'String',num2str(distances));
