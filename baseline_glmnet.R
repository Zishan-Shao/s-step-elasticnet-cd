library(glmnet)
file_path <- "/Users/shaozishan/Desktop/Research/coordinate descent/colon-cancer.txt"

# 初始化存储
y <- c()
X <- matrix(0, nrow = 0, ncol = 0)  # 初始化特征矩阵

# 打开文件
lines <- readLines(file_path)

# 解析每一行
for (i in seq_along(lines)) {
  line <- lines[i]
  parts <- strsplit(line, "\\s+")[[1]]  # 按空格分割
  
  # 提取目标变量 y（第一个元素），将浮点数转换为整数
  y_value <- as.numeric(parts[1])
  y <- c(y, ifelse(y_value == -1, 0, 1))  # 转换为 0 和 1
  
  # 初始化特征行
  if (nrow(X) == 0) {
    p <- max(as.numeric(sapply(parts[-1], function(kv) strsplit(kv, ":")[[1]][1])))  # 最大索引
    X <- matrix(0, nrow = length(lines), ncol = p)  # 初始化矩阵大小
  }
  
  # 解析特征部分
  kv_pairs <- parts[-1]  # 从第2个元素开始
  for (kv in kv_pairs) {
    # kv 形如 "2:-0.749474"
    sub_parts <- strsplit(kv, ":")[[1]]
    if (length(sub_parts) != 2) {
      stop(paste("解析 index:value 出错:", kv))
    }
    
    idx <- as.numeric(sub_parts[1])
    val <- as.numeric(sub_parts[2])
    
    # 放到矩阵 X 的对应位置
    if (idx < 1 || idx > ncol(X)) {
      stop(paste("特征索引超出范围:", idx))
    }
    X[i, idx] <- val
  }
}

# 检查结果
cat("目标变量 y:\n")
print(head(y))

cat("\n特征矩阵 X 的前 5 行:\n")
print(X[1:5, ])

# 保存数据到文件（可选）
# write.csv(X, "X.csv", row.names = FALSE)
# write.csv(y, "y.csv", row.names = FALSE)


# 合并 y 与 X
# 通常我们想要一个 data.frame, 第一列是 y, 后面是 X_1 ... X_p
df <- data.frame(
  y = y,
  X
)
colnames(df) <- c("y", paste0("X", 1:p))

# 查看前几行
head(df)

# 这里先演示下 biopy 数据
#data("biopsy", package = "MASS")
#df <- biopsy
# 移除有缺失的行
#df <- na.omit(df)

# 因为biopsy有一些列不是数值，这里先做简单处理
# biopsy默认最后一列class是benign/malignant，需要转成0/1
#df$class <- ifelse(df$class == "benign", 0, 1)
#df$ID <- NULL   # 去掉ID列
#head(df)

# 分割特征 X 和标签 y
X_all <- as.matrix(df[, -1])   # 最后一列是class
y_all <- df$y                    # 0/1

# 划分训练集和测试集(8:2)
# 注意：sample() 为简单做法，也可以使用caret等更加完善的方式
set.seed(42)
n_all <- nrow(X_all)
split_index <- floor(0.8 * n_all)  # 确定分割位置

# 按顺序划分数据
X_train <- X_all[1:split_index, , drop = FALSE]
y_train <- y_all[1:split_index]
X_test  <- X_all[(split_index + 1):n_all, , drop = FALSE]
y_test  <- y_all[(split_index + 1):n_all]

# 标准化(和 Python 里的 StandardScaler 类似)
# 可以使用 scale() 函数
X_train_scale <- scale(X_train)
# 测试集需要使用训练集的中心和标准差
train_center <- attr(X_train_scale, "scaled:center")
train_scale  <- attr(X_train_scale, "scaled:scale")
X_test_scale <- scale(X_test, center = train_center, scale = train_scale)
########################
## 8. 与 glmnet 对比
########################
cat("\n=== glmnet 对比 (alpha=0 => ridge, alpha=1 => lasso, etc.) ===\n")

# glmnet 要求 x 为矩阵/稀疏矩阵，y 可以是因子或者数值(0/1)；要 family="binomial" 做二分类
# 通常先要选 lambda，使用 cv.glmnet 做交叉验证
# 这里简化演示: 直接 fit glmnet，不做CV
t1 <- proc.time()
fit_glmnet <- glmnet(x = X_train_scale, 
                     y = y_train, 
                     alpha = 0.5,            # 0=>岭回归, 1=>Lasso, 可自己调整
                     family = "binomial",  # 逻辑回归
                     lambda = 0.2)
#maxit= 2000)        # 给个小的lambda
t2 <- proc.time() - t1
cat("glmnet fit 耗时(秒): ", t2[["elapsed"]], "\n")
# 提取并打印 beta
beta_glmnet <- coef(fit_glmnet)
cat("glmnet 模型系数 (beta):\n")
print(beta_glmnet[1:100])
# 预测
pred_glmnet_prob <- predict(fit_glmnet, newx = X_test_scale, type = "response")
pred_glmnet <- ifelse(pred_glmnet_prob >= 0.5, 1, 0)
acc_glmnet <- mean(pred_glmnet == y_test)
cat(sprintf("glmnet (alpha=0, lambda=0.01) 准确率: %.4f\n", acc_glmnet))
