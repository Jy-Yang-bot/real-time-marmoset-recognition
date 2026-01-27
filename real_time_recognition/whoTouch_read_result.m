function output = whoTouch_read_result()
    % read whoTouch_read_result - read YOLOv8 real-time detection results
    % output: 
    %   output - one cell that contains {name, time}
    %            name: most common name
    %            time: current system time

    % persist change，storing the most recent 30 detected results (labels)
    persistent recent_labels
    if isempty(recent_labels)
        recent_labels = {};  % initialize empty lists
    end

    % check if JSON exists
    if isfile('C:\\Users\\jyang291\\PycharmProjects\\MonkeyLogic_face_test\\whoTouch_weighted_detections.json')  % customize JSON file path
        % read JSON file
        data = fileread('C:\\Users\\jyang291\\PycharmProjects\\MonkeyLogic_face_test\\whoTouch_weighted_detections.json');
        result = jsondecode(data);
        
        % retrieve weighted_detections data list
        weighted_detections = result;
        
        % put weighted_detections content into recent_labels list
        recent_labels_add = [recent_labels, weighted_detections];
        
        % if list over 45, remove the oldest results
        if numel(recent_labels) > 45
            recent_labels_add(1:end-45) = [];  % keep the last 45 elements
        end
        
        % calculate the most common labels from the most recent 45 elements
        if ~isempty(recent_labels_add)
            counts = tabulate(recent_labels_add);
            [~, idx] = max(cell2mat(counts(:,2)));
            most_common_label = counts{idx,1};
        else
            most_common_label = 'no detection';
        end
        
        % retrieve current system time
        current_time = datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss');

        % output results
        output = {most_common_label, char(current_time)};
        disp(['Most common label in last 30 frames: ', most_common_label]);
        disp(['Time: ', char(current_time)]);
    else
        % if file not exists, output empty
        output = {[], []};
        disp('Waiting for detection results...');
    end
end