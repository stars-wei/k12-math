# k12-math

一个图谱驱动、可解释的 K12 数学求解原型：从 Neo4j `math` 图谱读取 Task、Strategy 和 Operation，使用注册过的 SymPy 操作执行策略路径，并输出经过验证的解题步骤。

`src/templates/result.html` 是页面源模板；每次求解生成的 `result.html` 是浏览器打开的结果文件。

`src/operation_registry.py` 将 Neo4j 的 `Operation.id` 映射到受信任的 SymPy 操作实现。

## 运行

```powershell
python -m pip install -r requirements.txt
python src/solve.py --expression "x**2/2 - 5*x + 1"
```

程序从项目根目录的 `.env` 读取 Neo4j 密码；未配置时会提示输入。随后生成 `result.html` 并自动在浏览器打开。页面使用 MathJax 在线渲染 LaTeX 公式。默认连接本机 Docker 容器暴露的 HTTP 地址：`http://localhost:7474/db/math/tx/commit`。

首次使用时复制环境变量示例文件：

```powershell
Copy-Item .env.example .env
```

然后在 `.env` 中填写本地凭据：

```dotenv
NEO4J_PASSWORD="你的 Neo4j 密码"
DEEPSEEK_API_KEY="你的 DeepSeek API Key"
SILICONFLOW_API_KEY="你的 SiliconFlow API Key"
```

`.env` 已被 Git 忽略，不要将真实密码或 API Key 写入 `.env.example`。

指定输出文件或不自动打开浏览器：

```powershell
python src/solve.py --expression "x**2/2 - 5*x + 1" --output answer.html --no-open
```

当前支持三个可执行 Task：

- `quadratic-function-axis`：求一元二次函数图像的对称轴。
- `quadratic-function-vertex`：求一元二次函数图像的顶点。
- `quadratic-function-extremum`：求一元二次函数的最大值或最小值。

使用配方法求最值：

```powershell
python src/solve.py `
  --expression "x**2 - 4*x + 3" `
  --task quadratic-function-extremum `
  --strategy extremum-completing-square
```

“配方法”本身只负责将一般式转化为顶点式。求对称轴和求最值分别通过组合 Strategy 复用它。

## 复合题目

网页会先识别题目中的全部题型，再逐项检查 Neo4j 中是否存在完整的可执行策略。一个题目可以同时包含多个要求；已入库的题型会执行策略，未找到可执行策略的题型会明确说明“本次未输出这部分答案”，不会影响其他题型继续求解。相同函数的顶点、对称轴和最值会复用已验证的中间事实，避免重复计算。

当前识别目录包括：

- 一元二次函数图像变换：可识别，尚未入库。
- 一元二次函数图像的顶点：可识别、可执行。
- 一元二次函数图像的对称轴：可识别、可执行。
- 一元二次函数变化趋势：可识别，尚未入库。
- 一元二次函数最大值或最小值：可识别、可执行。

题型识别目录与执行能力库相互独立。识别出题型并不代表系统已经具备该题型的 Strategy 和 Operation。

## 图谱迁移

首次使用 v0.2 时，在 Neo4j Browser 中执行 [`graph/migrations/v0.2_vertex_task.cypher`](graph/migrations/v0.2_vertex_task.cypher)。该迁移会新增“求顶点”的 Task 与策略路径，并为操作说明补充可由 MathJax 渲染的行内公式。

## 多函数题目

当公共题干要求对多个函数执行同一组任务时，表达式提取层会返回有序的 `items` 列表。系统按题号分组，并把识别出的公共题型分别应用到每一个函数。

例如“用配方法求下列两个函数的对称轴及最值”会生成四个求解任务：两个函数分别执行“配方法求对称轴”和“配方法求最值”。题干明确指定的方法优先于模型的策略选择，不会擅自改用公式法。

## 测试

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```
# 网页输入

在 PowerShell 中运行：

```powershell
python src/web.py
```

输入 Neo4j 密码后，打开 `http://127.0.0.1:8000`。网页可输入完整题干，也可上传 PNG/JPEG/WEBP 题目图片（最大 10 MB）。图片会先由 SiliconFlow 的 DeepSeek-OCR 转写，用户确认或修改题干后，DeepSeek 再提取 SymPy 表达式，并从 Neo4j 返回的候选 Task 与可执行 Strategy 中选择；选择结果和表达式都会由图谱与 SymPy 校验。

启动网页前，请确认 `.env` 已填写 `DEEPSEEK_API_KEY`；只有使用截图识别时才需要 `SILICONFLOW_API_KEY`。

可输入的完整题干例如：

```text
已知函数 y=(2x-1)^2+6(2x-1)+5，求图像的对称轴。
```
