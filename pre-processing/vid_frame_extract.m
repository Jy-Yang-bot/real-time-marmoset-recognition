% Read the video
dir = 'C:\Users\jyang291\Desktop\face_data\train01_20250108_13_19_04_Pro.mp4';
RawVid = VideoReader(dir);

% Frame rate (fps)
fps = 30;

% Calculate frame numbers for the time interval (39s to 41s)
start_frame = ceil(150 * fps);
end_frame = floor(194 * fps);

% Extract frames in the specified interval
for i = start_frame:6:end_frame
    % Read the current frame
    frame = read(RawVid, i);
    
    % Save the frame as an image
    imwrite(frame, ['C:\Users\jyang291\Desktop\mini_program_face_touchscreen\Dale\' int2str(i) '.jpg']);
end