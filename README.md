## ProcessTransformer:基于 Transformer 网络的预测性业务流程监控

![头部图片](https://github.com/Zaharah/processtransformer/blob/main/pt.JPG)
[![下载量](https://static.pepy.tech/badge/processtransformer)](https://pepy.tech/project/processtransformer)
[![下载量](https://static.pepy.tech/badge/processtransformer/month)](https://pepy.tech/project/processtransformer)
[![下载量](https://static.pepy.tech/badge/processtransformer/week)](https://pepy.tech/project/processtransformer)
<details><summary>摘要(点击展开)</summary>
<p>

预测性业务流程监控关注于使用事件日志预测正在运行的流程的未来特征。对流程执行过程的预判为高效运营、更好的资源管理以及有效的客户服务带来了巨大潜力。基于深度学习的方法已被广泛应用于流程挖掘中,以解决经典算法在处理多个问题时的局限性,尤其是下一事件和剩余时间预测任务。然而,设计一个在各种任务上都具有竞争力的深度神经架构颇具挑战,因为现有方法难以捕捉输入序列中的长程依赖,且在处理冗长的流程轨迹时表现不佳。在本文中,我们提出了 ProcessTransformer,这是一种利用基于注意力机制的网络从事件日志中学习高层表示的方法。我们的模型融合了长程记忆,并依赖自注意力机制来建立大量事件序列与相应输出之间的依赖关系。我们在九个真实事件日志上评估了该方法的适用性。我们证明,这个基于 Transformer 的模型在下一活动预测任务中平均取得了 80% 以上的准确率,优于先前技术的多个基线方法。与基线方法相比,我们的方法在预测正在运行案例的事件时间和剩余时间任务上也表现出了相当的竞争力。

</p>
</details>


#### 任务
- 下一活动预测(Next Activity Prediction)
- 下一活动时间预测(Time Prediction of Next Activity)
- 剩余时间预测(Remaining Time Prediction)

### 安装
```
pip install processtransformer
```
### 用法
我们提供了使用 ProcessTransformer 处理你自选事件日志所需的代码。我们以 helpdesk 数据集为例进行说明。

[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1tiOh2VS8yzOVON26CbmWn0oUn-dWAFhN?usp=sharing)

数据预处理请运行:

```python
python data_processing.py --dataset=helpdesk --task=next_activity
python data_processing.py --dataset=helpdesk --task=next_time
python data_processing.py --dataset=helpdesk --task=remaining_time
```

使用 ChronoTrace 的增强数据集时，传入原始 CSV 和对应的 `new_dataset.json`：

先在 `Accept_Course` 根目录的虚拟环境中注册本地 ChronoTrace 源码：

```python
python -m pip install -e 2026-06-20/ChronoTrace
```

```python
python data_processing.py --dataset=helpdesk_augmented --task=next_activity --raw_log_file=./datasets/raw/helpdesk/finale.csv --new_dataset=../../2026-06-20/ChronoTrace/outputs/ProcessTransformer/helpdesk/<run_id>/new_dataset.json
python data_processing.py --dataset=helpdesk_augmented --task=next_time --raw_log_file=./datasets/raw/helpdesk/finale.csv --new_dataset=../../2026-06-20/ChronoTrace/outputs/ProcessTransformer/helpdesk/<run_id>/new_dataset.json
python data_processing.py --dataset=helpdesk_augmented --task=remaining_time --raw_log_file=./datasets/raw/helpdesk/finale.csv --new_dataset=../../2026-06-20/ChronoTrace/outputs/ProcessTransformer/helpdesk/<run_id>/new_dataset.json
```
训练并评估模型请运行:

```python
python next_activity.py --dataset=helpdesk --epochs=100
python next_time.py --dataset=helpdesk --epochs=100
python remaining_time.py --dataset=helpdesk --epochs=100
```


### 工具
- <a href="http://tensorflow.org/">Tensorflow >=2.4</a>

## 数据
用于预测性业务流程监控的事件日志可在 [4TU Research Data](https://data.4tu.nl/categories/_/13500?categories=13503) 获取。

## 如何引用

如果你使用了本项目的代码或思路,请考虑引用我们的论文:

Zaharah A. Bukhsh, Aaqib Saeed, & Remco M. Dijkman. (2021). ["ProcessTransformer: Predictive Business Process Monitoring with Transformer Network"](https://arxiv.org/abs/2104.00721). arXiv preprint arXiv:2104.00721


```
@misc{bukhsh2021processtransformer,
      title={ProcessTransformer: Predictive Business Process Monitoring with Transformer Network}, 
      author={Zaharah A. Bukhsh and Aaqib Saeed and Remco M. Dijkman},
      year={2021},
      eprint={2104.00721},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## 在 macOS(Apple Silicon,仅 CPU)上复现

原始代码面向 TensorFlow 2.4 以及一套数年前的依赖版本。在一台现代的、仅使用 CPU 的 Mac(Apple Silicon,Python 3.12)上,你需要更新的技术栈以及一些兼容性调整。以下是经过验证的配置方法。

### 环境配置

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

- `requirements.txt` 锁定了经过验证的依赖组合,无需手动逐个指定包名,直接 `pip install -r requirements.txt` 即可。相比 `setup.py` 里过于宽松的约束(如 `tensorflow>=2.4`,会拉到最新版而触发兼容性问题),它明确锁定 `tensorflow==2.17.*` 和 `tf-keras`,在 Apple Silicon + Python 3.9–3.12 上稳定可用。
- `tensorflow==2.17.*` 提供了 Apple Silicon 原生的 arm64 wheel,可在 CPU 上运行。
- 代码使用的是旧版 `tf.keras`(Keras 2)API。TF 2.16+ 默认使用 Keras 3,因此需要安装 `tf-keras` 包,并在每次运行前设置环境变量 `TF_USE_LEGACY_KERAS=1`:

```bash
export TF_USE_LEGACY_KERAS=1
```

> **Python 版本提示**:`requirements.txt` 中的 `tensorflow==2.17.*` 需要 Python 3.9–3.12。如果新机器是 Python 3.13,TF 2.17 尚无对应 wheel,请先用 3.9–3.12 建立虚拟环境。

### 本版本已应用的兼容性修复

`processtransformer/data/processor.py` 和 `data_processing.py` 已经过修补,使其能够在当前版本的 `pandas` / `numpy` / `multiprocessing` 上运行:

- `_load_df`:时间戳解析从 `pd.to_datetime(dayfirst=True)`(在 pandas >= 2.0 上会失败)改为 `format="mixed", utc=True`,可同时兼容 helpdesk 的 `YYYY/MM/DD` 格式和带时区的 XES ISO-8601 时间戳。
- 所有 `np.where(cond, a, b)` 标量赋值写法都替换为 Python 三元表达式(`a if cond else b`),因为新版 numpy 返回的是 ndarray 而非标量,会破坏 `DataFrame.at` 的索引。
- 当 `pool <= 1` 时,`multiprocessing.Pool` 被替换为顺序执行(通过新增的 `_run_pool` 辅助方法),因为 macOS 的 Python 3.8+ 使用 `spawn` 启动方式,这会让单进程的 `Pool` 变慢并静默吞掉异常。
- `data_processing.py` 现在接受 `--columns` 参数(`helpdesk` 预设、`xes` 预设,或逗号分隔的 `case,activity,time` 三元组),这样无需修改代码即可处理 XES 衍生的日志。

### 数据预处理

原始日志位于 `datasets/raw/` 下。helpdesk 日志使用 helpdesk 风格的列名;所有 XES 衍生的日志(bpic*、hospital、traffic_fines)使用 `case:concept:name` / `concept:name` / `time:timestamp`。

```bash
export TF_USE_LEGACY_KERAS=1
PY=.venv/bin/python

# helpdesk(helpdesk 列名预设)
$PY data_processing.py --dataset=helpdesk --task=next_activity  --raw_log_file=./datasets/raw/helpdesk/finale.csv
$PY data_processing.py --dataset=helpdesk --task=next_time      --raw_log_file=./datasets/raw/helpdesk/finale.csv
$PY data_processing.py --dataset=helpdesk --task=remaining_time --raw_log_file=./datasets/raw/helpdesk/finale.csv

# XES 衍生日志(需加 --columns=xes)
$PY data_processing.py --dataset=bpic2012 --task=next_activity --columns=xes \
    --raw_log_file="./datasets/raw/bpic2012/BPI_Challenge_2012.csv"
```

对每个数据集和每个任务重复上述步骤。处理后的文件会写入 `datasets/<dataset>/processed/`。

### 训练与评估

论文使用 100 个 epoch 和 1e-2 的学习率(见第 5.2 节)。脚本的默认值(10 个 epoch,学习率 0.001)与论文**不一致**,因此需要显式传入这些参数:

```bash
$PY next_activity.py  --dataset=helpdesk --epochs=100 --batch_size=12 --learning_rate=0.01
$PY next_time.py      --dataset=helpdesk --epochs=100 --batch_size=12 --learning_rate=0.01
$PY remaining_time.py --dataset=helpdesk --epochs=100 --batch_size=12 --learning_rate=0.01
```

结果保存在 `results/<dataset>/results_<task>.csv` 中;最后一行是跨所有前缀长度的平均指标,对应论文中的表 2。

### CPU 上的预期运行时间

| 数据集 | 案例数 | 事件数 | 每个 epoch 大约耗时(next_activity) |
|---|---|---|---|
| helpdesk | 4,580 | 21,348 | ~4 秒 |
| bpic2020_domestic | 10,500 | 56,437 | 数分钟 |
| bpic2013_incidents | 7,554 | 65,533 | 数分钟 |
| bpic2012 | 13,087 | 262,200 | 数十分钟 |
| bpic2020_international | 6,449 | 72,151 | 数十分钟 |
| hospital_billing | 99,999 | 451,359 | 单任务数小时 |
| traffic_fines | 150,370 | 561,470 | 单任务数小时 |

建议先用 helpdesk 验证整个流水线,再逐步扩大规模。

### 数据集覆盖情况与论文的对比

论文在 9 个事件日志上进行了评估(表 1)。本版本包含 7 个。缺失的两个变体 **BPIC12w** 和 **BPIC12cw** 是 BPIC2012 的子集(根据论文参考文献 [7],按生命周期转换 / concept:name 过滤);如需复现它们,可按相应规则过滤 `bpic2012`。在 helpdesk 上经过验证的冒烟测试在 2 个 epoch 后达到了 85.19% 的准确率,接近论文报告的 85.63%。
