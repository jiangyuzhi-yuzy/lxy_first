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

% 初始化变量（根据要求设置初值）
current_m1 = 450;  % m1初值
current_d1 = 250;  % d1初值
current_d2 = 550;  % d2初值

% 初始化当前目标函数值和初始函数值
current_s = s(current_m1, current_d1, current_d2);
current_s_init = s_init(current_m1, current_d1, current_d2);

% 初始化最佳结果变量（不使用结构体，直接用变量存储）
best_m1 = current_m1;
best_d1 = current_d1;
best_d2 = current_d2;
best_s_value = current_s;
best_s_init_value = current_s_init;
best_outer_iter = 1;
best_inner_iter = 0;

% 定义迭代参数
max_outer_iter = 50;  % 外循环最高迭代50次
tolerance = 1e-13;    % 收敛条件

% 定义搜索范围
d1_min = 20;
d1_max = 1000;
d2_min = 20;
d2_max = 1000;
m1_min = 324;         % m1范围
m1_max = 500;

% 打印初始信息
fprintf('初始参数: m1=%d, d1=%d, d2=%d\n', current_m1, current_d1, current_d2);
fprintf('初始目标函数值s: %.8f\n', current_s);
fprintf('初始函数值s_init: %.8f\n\n', current_s_init);

% 块坐标下降外循环
outer_iter = 1;
while outer_iter <= max_outer_iter
    % 打印外循环信息，包含当前初始函数值
    fprintf('外循环迭代次数: %d, 当前m1: %d, 当前s值: %.8f, 当前s_init值: %.8f\n', ...
            outer_iter, current_m1, current_s, current_s_init);
    
    % 保存当前状态用于收敛判断
    prev_s = current_s;
    prev_s_init = current_s_init;
    prev_m1 = current_m1;
    
    % 遍历所有可能的m1值寻找更优点
    best_m1_candidate = current_m1;
    best_m1_s = current_s;
    best_m1_s_init = current_s_init;
    
    for m1_candidate = m1_min:m1_max
        % 检查该m1是否能满足约束条件（使用当前d1, d2）
        if check_constraints(m1_candidate, current_d1, current_d2)
            % 计算该m1对应的目标函数值和初始函数值
            candidate_s = s(m1_candidate, current_d1, current_d2);
            candidate_s_init = s_init(m1_candidate, current_d1, current_d2);
            
            % 如果找到更优的m1
            if candidate_s < best_m1_s
                best_m1_candidate = m1_candidate;
                best_m1_s = candidate_s;
                best_m1_s_init = candidate_s_init;
            end
        end
    end
    
    % 更新m1及对应的函数值
    current_m1 = best_m1_candidate;
    current_s = best_m1_s;
    current_s_init = best_m1_s_init;
    
    % 优化d1和d2（内循环）
    k_iter = 1;
    max_inner_iter = 1000;
    inner_converged = false;
    
    while k_iter < max_inner_iter
        % 打印内循环信息，包含当前初始函数值
        fprintf('  内循环迭代次数: %d, 当前d1: %d, 当前d2: %d, 当前s值: %.8f, 当前s_init值: %.8f\n', ...
                k_iter, current_d1, current_d2, current_s, current_s_init);
        
        % 先优化d2，固定d1
        best_d2 = find_best_d2(current_m1, current_d1, d2_min, d2_max);
        current_d2 = best_d2;
        
        % 再优化d1，固定d2
        best_d1 = find_best_d1(current_m1, current_d2, d1_min, d1_max);
        current_d1 = best_d1;
        
        % 计算新的目标函数值和初始函数值
        s_k = s(current_m1, current_d1, current_d2);
        s_init_k = s_init(current_m1, current_d1, current_d2);
        
        % 检查内循环收敛条件
        if abs(s_k - current_s) <= 1e-12
            current_s = s_k;
            current_s_init = s_init_k;
            inner_converged = true;
            break;
        else
            current_s = s_k;
            current_s_init = s_init_k;
            k_iter = k_iter + 1;
        end
    end
    
    % 显示内循环最终结果
    fprintf('  内循环结束，迭代次数: %d, 更新后s值: %.8f, 更新后s_init值: %.8f\n', ...
            k_iter, current_s, current_s_init);
    
    % 更新最佳结果
    if current_s < best_s_value
        best_m1 = current_m1;
        best_d1 = current_d1;
        best_d2 = current_d2;
        best_s_value = current_s;
        best_s_init_value = current_s_init;
        best_outer_iter = outer_iter;
        best_inner_iter = k_iter;
    end
    
    % 检查外循环收敛条件
    if abs(current_s - prev_s) <= tolerance
        fprintf('外循环在第%d次迭代收敛\n', outer_iter);
        break;
    end
    
    outer_iter = outer_iter + 1;
end

% 输出最佳结果
fprintf('\n========================================\n');
fprintf('全局最优点信息：\n');
fprintf('m1: %d\n', best_m1);
fprintf('最优d1: %d\n', best_d1);
fprintf('最优d2: %d\n', best_d2);
fprintf('最小目标函数值s: %.8f\n', best_s_value);
fprintf('对应的初始函数值s_init: %.8f\n', best_s_init_value);
fprintf('达到收敛的外循环迭代次数: %d\n', best_outer_iter);
fprintf('最后一次内循环迭代次数: %d\n', best_inner_iter);
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


% 检查约束条件（内部逻辑不变）
function result = check_constraints(m1, d1, d2)
    global sab_forward se1_forward sba_backward se2_backward M;
    
    % 计算各概率值
    sab = sab_forward(m1, d1);
    se1 = se1_forward(m1, d1);
    sba = sba_backward(m1, d2);
    se2 = se2_backward(m1, d2);
    
    % 检查所有约束条件（添加数值稳定性处理，避免极端值）
    if sab > 0.5 || sab <= 1e-10  % sab需在(1e-10, 0.5)范围内
        result = false;
        return;
    end
    
    if se1 < 0.5 || se1 >= 1 - 1e-10  % se1需在[0.5, 1-1e-10)范围内
        result = false;
        return;
    end
    
    if sba > 0.5 || sba <= 1e-10  % sba需在(1e-10, 0.5)范围内
        result = false;
        return;
    end
    
    if se2 < 0.5 || se2 >= 1 - 1e-10  % se2需在[0.5, 1-1e-10)范围内
        result = false;
        return;
    end
    
    result = true;
end