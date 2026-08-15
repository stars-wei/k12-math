// v0.2: give “求顶点” an executable graph-backed path and make inline maths renderable.

MERGE (task:Task {id: 'quadratic-function-vertex'})
SET task += {
  id: 'quadratic-function-vertex',
  name: '求一元二次函数图像的顶点',
  description: '通过配方法得到顶点式，并读取顶点坐标。'
}
MERGE (strategy:Strategy {id: 'vertex-by-completing-square'})
SET strategy += {
  id: 'vertex-by-completing-square',
  name: '配方法求顶点',
  description: '将函数配成顶点式后读取顶点信息。'
}
MERGE (task)-[:USES]->(strategy)
WITH strategy
MATCH (complete:Strategy {id: 'axis-completing-square'})
MATCH (read:Strategy {id: 'axis-vertex-read'})
MATCH (back:Operation {id: 'op-substitute-back-linear'})
MERGE (strategy)-[:USES {order: 1}]->(complete)
MERGE (strategy)-[:USES {order: 2}]->(read)
MERGE (strategy)-[:USES {order: 3, condition_id: 'substitution-active'}]->(back);

MATCH (op:Operation {id: 'op-calculate-axis-formula'})
SET op.display_description = '根据对称轴公式 \\(x=-\\frac{b}{2a}\\) 计算对称轴。';

MATCH (op:Operation {id: 'op-rewrite-linear-coefficient'})
SET op.display_description = '将一次项系数写成 \\(2p\\) 的形式，其中 \\(p=\\frac{b}{2a}\\)。';

MATCH (op:Operation {id: 'op-complete-square-term'})
SET op.display_description = '在括号内同时加上并减去 \\(\\left(\\frac{b}{2a}\\right)^2\\)。';

MATCH (op:Operation {id: 'op-read-vertex-form'})
SET op.display_description = '从顶点式 \\(y=a(x-h)^2+k\\) 读取顶点、对称轴与顶点函数值。';

MATCH (op:Operation {id: 'op-determine-extremum-by-sign'})
SET op.display_description = '由 \\(a\\) 的正负判断抛物线开口方向，进而确定最值类型。';
