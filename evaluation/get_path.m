function v = get_path(arg)
%   ��ȡtest dataset���
% 	Input       arg     ���� in ['tri', 'sep']
%                       interpolate��tri, ����������sep.

if strcmp(arg, 'tri')
	v = get_path_tri();
end

if strcmp(arg, 'sep')
	v = get_path_sep();
end

end