% data_path = '/home/ftp/Coldog/Dataset/toflow/vimeo_septuplet/sequences';
% noise_path = '/home/ftp/Coldog/Dataset/toflow/vimeo_septuplet/sequences_down';
data_path = 'F:\Dataset\vimeo_triplet\tiny\test_blur';
noise_path = 'F:\Dataset\vimeo_triplet\tiny\test_down';
if ~exist(noise_path, 'dir')
  mkdir(noise_path);
end
videos = dir(data_path);
% total_num = 91701;
total_num = 37;
count = 0;
disp('Start downsampling...')
for i=1:length(videos)
    if strcmp(videos(i).name, '.') || strcmp(videos(i).name, '..')
        continue
    end
    seps = dir(fullfile(data_path, videos(i).name));
    for j=1:length(seps)
        if strcmp(seps(j).name, '.') || strcmp(seps(j).name, '..')
            continue
        end
        downsample_input(fullfile(data_path, videos(i).name, seps(j).name), fullfile(noise_path, videos(i).name, seps(j).name))
        count = count + 1;
        if mod(count,5)==0
            disp(['  Processed ',num2str(count/total_num*100), '%'])
        end
    end
end
disp(['  Processed ',num2str(count/total_num*100), '%'])