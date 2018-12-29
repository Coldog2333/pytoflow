function result = evaluate(output_dir, target_dir, task)
% result = evaluate(output_dir, target_dir, task)
% evaluate output_dir的out.png和target_dir的im2.png
%   Input   output_dir  输出图片的目录 
%           target_dir  reference图片的目录
% 
  ts = get_task(output_dir, target_dir, task);
  [p, s, a] = run_eval_template(ts{1,2},ts{1,3},ts{1,4},get_path(ts{1,5}),ts{1,6},ts{1,7});
  % Input
  %     Number of samples
  %     target dir
  %     output dir
  %     测试集所在目录号
  %     name of target figure
  %     name of output figure
  result = [ts{1,1} ' psnr,ssim,abs= ' num2str(p) ', ' num2str(s) ', ' num2str(a)];
end
