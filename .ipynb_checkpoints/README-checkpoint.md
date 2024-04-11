### 项目结构

1. 当前目录：

   - Qwen1.5_14B_Psychat_Train.ipynb

   - Qwen1.5_14B_Psychat_Predict_webui.ipynb

   以上是直接利用 200 条中文数据集，微调 Qwen2-14 对应代码，由于平台限制，其数据集在外侧，对应关系如下：

   - data-1：中文数据集
   - data-2：Qwen 模型参数
   - data-3：英文数据集
   - pretrain：利用 data-1，data-2 训练得到的demo1

2. build_psymodel:

   - train1_Chinese.ipynb
   - train2.ipynb
   - webui.ipynb

   以上可以看作 demo2，是之前的推倒重来，在上述所做之前，

   - 加上近10万条的英文数据训练，即 trian2 所作，
   - 然后 train1_Chinese 等同于 demo1 中的训练，只是多加了一步 加载 train2 得到的参数，
   - 最后 webui 等同于 demo1 中的预测，也只是多加载了一部分参数。
   - train1 中文的参数在 `build_psymodel/C_output`
   - train2 英文的参数在 `build_psymodel/E_output`

3. run

   用来运行程序，之后要加上 demo2 的运行。