% 固定参数设置
M = 1000;
current_ye1 = 0.4;
current_ye2 = 0.6;
current_yab = 0.9;
current_yba = 1.0;

% 辅助函数定义
V = @(y) 1 - (1 + y).^(-2);
C = @(y) log2(1 + y);

% 计算当前信道参数
Ve1 = V(current_ye1);
Ce1 = C(current_ye1);
Vab = V(current_yab);
Cab = C(current_yab);
Ve2 = V(current_ye2);
Ce2 = C(current_ye2);
Vba = V(current_yba);
Cba = C(current_yba);

% 定义概率函数
sab_forward = @(m1, d1) 1 - normcdf(log(2)*sqrt(m1/Vab).*(Cab - d1./m1));
se1_forward = @(m1, d1) 1 - normcdf(log(2)*sqrt(m1/Ve1).*(Ce1 - d1./m1));
sba_backward = @(m1, d2) 1 - normcdf(log(2)*sqrt((M-m1)/Vba).*(Cba - d2./(M-m1)));
se2_backward = @(m1, d2) 1 - normcdf(log(2)*sqrt((M-m1)/Ve2).*(Ce2 - d2./(M-m1)));

% 目标函数定义
s_init = @(m1, d1, d2) 1 - (1 - sab_forward(m1, d1)) .* se1_forward(m1, d1) .* ...
    (1 - sba_backward(m1, d2)) .* se2_backward(m1, d2);
s = @(m1, d1, d2) 1/(1 - sab_forward(m1, d1)) .* 1/(se1_forward(m1, d1)) .* ...
    1/((1 - sba_backward(m1, d2))) .* 1/(se2_backward(m1, d2));

% 凸近似函数定义
s_maxmize_op = @(m1, d1, d2) (1/(1 - sab_forward(m1, d1)) + 1/(se1_forward(m1, d1)) + ...
    1/((1 - sba_backward(m1, d2))) + 1/(se2_backward(m1, d2)))^2 / 16;

% 声明全局变量，用于子函数访问
global sab_forward se1_forward sba_backward se2_backward s_maxmize_op;
global current_s current_d1 current_d2 M;

% 初始化变量
current_d1 = 250;
current_d2 = 500;

% 定义迭代参数
max_iter = 1000;
tolerance = 1e-16;

% 定义搜索范围
d1_min = 100;
d1_max = 1000;
d2_min = 20;
d2_max = 1000;

% 创建结果存储结构
results = struct('m1', [], 'd1', [], 'd2', [], 's_value', [], 's_init_value', [], 'iterations', []);

% 遍历m1从324到500的每个整数
for m1 = 324:500
    current_m1 = m1;
    % 初始化当前s值
    current_s = s(current_m1, current_d1, current_d2);
    k_iter = 1;
    
    % 优化d1和d2
    while k_iter < max_iter
        % 先优化d2，固定d1
        best_d2 = find_best_d2(current_m1, current_d1, d2_min, d2_max);
        current_d2 = best_d2;
        
        % 再优化d1，固定d2
        best_d1 = find_best_d1(current_m1, current_d2, d1_min, d1_max);
        current_d1 = best_d1;
        
        % 计算新的目标函数值
        s_k = s(current_m1, current_d1, current_d2);
        
        % 检查收敛条件
        if abs(s_k - current_s) <= tolerance
            break;
        else
            current_s = s_k;
            k_iter = k_iter + 1;
        end
    end
    
    % 存储结果
    results(end+1).m1 = current_m1;
    results(end).d1 = current_d1;
    results(end).d2 = current_d2;
    results(end).s_value = s(current_m1, current_d1, current_d2);
    results(end).s_init_value = s_init(current_m1, current_d1, current_d2);
    results(end).iterations = k_iter;
    
    % 显示进度
    fprintf('完成m1 = %d的优化，迭代次数: %d\n', current_m1, k_iter);
end

% 找出全局最优点（目标函数s值最小的点）
min_s_value = Inf;
global_opt_index = 0;

for i = 1:length(results)
    if results(i).s_value < min_s_value
        min_s_value = results(i).s_value;
        global_opt_index = i;
    end
end

% 输出所有结果
fprintf('\n所有m1值的优化结果：\n');
fprintf('m1\t最优d1\t最优d2\t目标函数值s\t初始目标函数值s_init\t迭代次数\n');
for i = 1:length(results)
    fprintf('%d\t%d\t%d\t%.8f\t%.8f\t%d\n', ...
        results(i).m1, ...
        results(i).d1, ...
        results(i).d2, ...
        results(i).s_value, ...
        results(i).s_init_value, ...
        results(i).iterations);
end

% 输出全局最优点
fprintf('\n========================================\n');
fprintf('全局最优点信息：\n');
fprintf('m1: %d\n', results(global_opt_index).m1);
fprintf('最优d1: %d\n', results(global_opt_index).d1);
fprintf('最优d2: %d\n', results(global_opt_index).d2);
fprintf('最小目标函数值s: %.8f\n', results(global_opt_index).s_value);
fprintf('对应的初始目标函数值s_init: %.8f\n', results(global_opt_index).s_init_value);
fprintf('达到收敛的迭代次数: %d\n', results(global_opt_index).iterations);
fprintf('========================================\n');


% 寻找最优d2的函数（固定d1）
function best_d2 = find_best_d2(m1, d1, d2_min, d2_max)
    global sab_forward se1_forward sba_backward se2_backward s_maxmize_op;
    global current_s current_d1 current_d2;
    
    best_val = Inf;
    best_d2 = d2_min;
    
    % 遍历所有可能的整数d2值
    for d2 = d2_min:d2_max
        % 检查约束
        if ~check_constraints(m1, d1, d2)
            continue;
        end
        
        % 计算目标函数值
        val = s_maxmize_op(m1, d1, d2) + (current_s - s_maxmize_op(m1, current_d1, current_d2));
        
        % 更新最优值
        if val < best_val
            best_val = val;
            best_d2 = d2;
        end
    end
end

% 寻找最优d1的函数（固定d2）
function best_d1 = find_best_d1(m1, d2, d1_min, d1_max)
    global sab_forward se1_forward sba_backward se2_backward s_maxmize_op;
    global current_s current_d1 current_d2;
    
    best_val = Inf;
    best_d1 = d1_min;
    
    % 遍历所有可能的整数d1值
    for d1 = d1_min:d1_max
        % 检查约束
        if ~check_constraints(m1, d1, d2)
            continue;
        end
        
        % 计算目标函数值
        val = s_maxmize_op(m1, d1, d2) + (current_s - s_maxmize_op(m1, current_d1, current_d2));
        
        % 更新最优值
        if val < best_val
            best_val = val;
            best_d1 = d1;
        end
    end
end

% 检查约束条件
function result = check_constraints(m1, d1, d2)
    global sab_forward se1_forward sba_backward se2_backward M;
    
    % 计算各概率值
    sab = sab_forward(m1, d1);
    se1 = se1_forward(m1, d1);
    sba = sba_backward(m1, d2);
    se2 = se2_backward(m1, d2);
    
    % 检查所有约束条件（添加数值稳定性处理）
    if sab > 0.5 || sab <= 1e-10  % 避免接近0的值
        result = false;
        return;
    end
    
    if se1 < 0.5 || se1 >= 1 - 1e-10  % 避免接近1的值
        result = false;
        return;
    end
    
    if sba > 0.5 || sba <= 1e-10
        result = false;
        return;
    end
    
    if se2 < 0.5 || se2 >= 1 - 1e-10
        result = false;
        return;
    end
    
    result = true;
end
