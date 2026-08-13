# math-v0.1

最小可行版本：从 Neo4j `math` 图谱读取 Task、Strategy 和 Operation，使用注册过的 SymPy 操作执行策略路径，并输出经过验证的解题步骤。

`templates/result.html` 是页面源模板；每次求解生成的 `result.html` 是浏览器打开的结果文件。

`operation_registry.py` 将 Neo4j 的 `Operation.id` 映射到受信任的 SymPy 操作实现。

## 运行

```powershell
python -m pip install -r requirements.txt
python solve.py --expression "x**2/2 - 5*x + 1"
```

程序会提示输入 Neo4j 密码，生成 `result.html` 并自动在浏览器打开。页面使用 MathJax 在线渲染 LaTeX 公式。默认连接本机 Docker 容器暴露的 HTTP 地址：`http://localhost:7474/db/math/tx/commit`。

也可以先在终端设置密码，避免每次输入：

```powershell
$env:NEO4J_PASSWORD = '你的 Neo4j 密码'
```

指定输出文件或不自动打开浏览器：

```powershell
python solve.py --expression "x**2/2 - 5*x + 1" --output answer.html --no-open
```

当前支持两个 Task：

- `quadratic-function-axis`：求一元二次函数图像的对称轴。
- `quadratic-function-extremum`：求一元二次函数的最大值或最小值。

使用配方法求最值：

```powershell
python solve.py `
  --expression "x**2 - 4*x + 3" `
  --task quadratic-function-extremum `
  --strategy extremum-completing-square
```

“配方法”本身只负责将一般式转化为顶点式。求对称轴和求最值分别通过组合 Strategy 复用它。

## 复合题目

网页会先识别题目中的全部题型，再逐项检查 Neo4j 中是否存在完整的可执行策略。一个题目可以同时包含多个要求；已入库的题型会执行策略，未找到可执行策略的题型会显示“该题型未入库”，不会影响其他题型继续求解。

当前识别目录包括：

- 一元二次函数图像变换：可识别，尚未入库。
- 一元二次函数图像的对称轴：可识别、可执行。
- 一元二次函数变化趋势：可识别，尚未入库。
- 一元二次函数最大值或最小值：可识别、可执行。

题型识别目录与执行能力库相互独立。识别出题型并不代表系统已经具备该题型的 Strategy 和 Operation。

## 多函数题目

当公共题干要求对多个函数执行同一组任务时，表达式提取层会返回有序的 `items` 列表。系统按题号分组，并把识别出的公共题型分别应用到每一个函数。

例如“用配方法求下列两个函数的对称轴及最值”会生成四个求解任务：两个函数分别执行“配方法求对称轴”和“配方法求最值”。题干明确指定的方法优先于模型的策略选择，不会擅自改用公式法。

## 测试

```powershell
python -m unittest -v test_solve.py
```
# 网页输入

在 PowerShell 中运行：

```powershell
python web.py
```

输入 Neo4j 密码后，打开 `http://127.0.0.1:8000`。网页可输入完整题干，也可上传 PNG/JPEG/WEBP 题目图片（最大 10 MB）。图片会先由 SiliconFlow 的 DeepSeek-OCR 转写，用户确认或修改题干后，DeepSeek 再提取 SymPy 表达式，并从 Neo4j 返回的候选 Task 与可执行 Strategy 中选择；选择结果和表达式都会由图谱与 SymPy 校验。

启动网页前，设置 DeepSeek 密钥：

```powershell
$env:DEEPSEEK_API_KEY = '你的 DeepSeek API Key'
$env:SILICONFLOW_API_KEY = '你的 SiliconFlow API Key'
```

可输入的完整题干例如：

```text
已知函数 y=(2x-1)^2+6(2x-1)+5，求图像的对称轴。
```
