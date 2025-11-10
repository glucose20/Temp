# LLMDTA

LLMDTA: 使用生物序列大模型改进药物-靶点亲和力的冷启动预测。

![](./LLMDTA-model.png)

## 需求

gensim==4.3.1 \
matplotlib==3.2.2 \
mol2vec==0.1 \
numpy==1.23.4 \
pandas==1.5.2 \
rdkit==2023.3.2 \
scikit_learn==1.2.2 \
scipy==1.8.1 \
torch==1.8.2 \
tqdm==4.65.0 \

从[ESM2](https://github.com/facebookresearch/esm)仓库安装。

## 资源

- code
  - train.py: 训练脚本。
  - pred.py: 推理脚本。
  - hyperparameter.py: 训练配置。
  - hyperparameter4pred.py: 推理配置。
  - ...
- code_prepareEmb
  - \_Split5FoldDataset.ipynb: 划分 5 折交叉数据集。
  - \_PreparePretrain.ipynb: 在训练阶段，计算并保存预训练特征。
- data: 放置训练、推理数据集
- savemodel: 放置推理模型。

## 示例用法

### 使用 Davis\KIBA\Metz 数据集进行训练

1. 从这个[Kaggle Dataset](https://www.kaggle.com/datasets/christang0002/llmdta/data)下载数据集。

2. 配置 `hyperparameter.py`

```
self.data_root       : 数据集存放路径
self.dataset         : 数据集名称, davis, kiba or metz
self.running_set     : 冷启动设置： warm, novel-drug, novel-prot or novel-pair

self.mol2vec_dir    : 提取的药物预训练特征路径
self.protvec_dir    : 提取的蛋白质预训练特征路径
self.drugs_dir      : 药物列表：[drug_id, drug_seq]
self.prots_dir      : 蛋白质列表: [prot_id, prot_seq]

self.cuda           : 计算设备
```

3. 运行 `python code/train.py`

### 使用自定义数据集进行训练

如果你想使用 LLMDTA 在你的数据集上进行训练，你应该首先准备数据集和预训练特征。

在 code_prepareEmb 文件夹中，我们提供了生成预训练嵌入和划分 5 折冷启动数据集的笔记本。

1. 准备自定义数据集。\
   我们期望训练数据使用以下格式：

- `drugs.csv`，
  ['drug_id', 'drug_smile']
- `targets.csv`，['prot_id','prot_seq']
- `pairs.csv`，['drug_id', 'prot_id', 'label']

2. 生成预训练嵌入。\
   使用 `code_prepareEmb/_PreparePretrain.ipynb` 从 drugs.csv 和 targets.csv 中提取 mol2vec 和 ESM2 预训练特征。

3. 配置 `hyperparameter.py`

4. 运行 `python code/train.py`

### 推理

1. 准备类似 CSV 格式的药物和靶标。\
   这里提供两种推理方式：

- 用户直接提供药物-靶标数据对，[drug_id, prot_id, smiles, prot_seq]
- 用户提供药物和靶标列表，推理脚本自动计算所有两两组合成的数据对，并预测。
  - drug.csv, [drug_id, drug_smile]
  - prot.csv, [prot_id, prot_seq]

2. 配置 `hyperparameter4pred.py` 文件。

```
self.pred_dataset   : 预测任务名称, 用于保存结果命名
self.sep : 读取药物/靶标列表时的分隔符

self.pred_pair_pth  : 推理方式1中，药物靶标对文件位置

self.pred_drug_dir : 方式2，药物列表文件
self.pred_prot_dir : 方式2，靶标列表文件

self.model_fromTrain : 预训练模型

```

3. 运行预测脚本。结果文件将保存在当前路径。

```
python code/pred.py
```
