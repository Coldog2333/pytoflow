function v = get_path(arg)
%   获取test dataset编号
% 	Input       arg     参数 in ['tri', 'sep']
%                       interpolate用tri, 其余两个用sep.

if strcmp(arg, 'tri')
	v = get_path_tri();
end

if strcmp(arg, 'sep')
	v = get_path_sep();
end

end