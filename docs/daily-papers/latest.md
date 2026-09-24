# 最近更新

<div class="paper-overview">
<div class="paper-overview__item"><span>更新日期</span><strong><a href="../2026-09-23/">2026-09-23</a></strong></div>
<div class="paper-overview__item"><span>论文总数</span><strong>10</strong></div>
<div class="paper-overview__item"><span>数据接口</span><strong><a href="../../api/latest.json">latest.json</a></strong></div>
</div>

## 来源概览

<div class="paper-source-grid">
<div class="paper-source-card">
<span>arXiv</span>
<strong>10 篇</strong>
<a href="../2026-09-23-arxiv/">查看来源页面</a>
</div>
<div class="paper-source-card">
<span>bioRxiv</span>
<strong>0 篇</strong>
<a href="../2026-09-23-biorxiv/">查看来源页面</a>
</div>
<div class="paper-source-card">
<span>medRxiv</span>
<strong>0 篇</strong>
<a href="../2026-09-23-medrxiv/">查看来源页面</a>
</div>
</div>

## 当期论文

<div class="paper-list">
<article class="paper-item">
<h3>用流匹配改进集合滤波器</h3>
<span class="paper-item__meta">arXiv / 2026-09-23 / Haoyuan Chen</span>
<p>数据同化从部分且含噪的观测中估计动力状态。经典集合滤波器效率高，但通过有限样本协方差和仿射高斯分布限制了分析更新。我们引入流集合滤波器（FlowEF），它使用条件流匹配将预报集合从经典基线滤波器输运到分析集合。FlowEF在训练时使用局部化高斯源，在部署时输运来自基线滤波器的预报集合成员，并将其速度场条件化于来自该基线滤波器的集合和观测。因此，所提出的模型学习非线性更新，同时独立地映射每个基线集合。对于稀疏观测的动力系统，FlowEF在确定性和概率性指标上均优于所有四种经典集合滤波器。它还在最先进的生成式数据同化模型中取得了最佳性能。</p>
<div class="paper-item__links"><a href="https://arxiv.org/pdf/2609.28015v1">PDF</a><a href="http://arxiv.org/abs/2609.28015v1">论文页面</a></div>
</article>
<article class="paper-item">
<h3>SoLiD26：用于机器学习原子间势的第一性原理固液界面数据集</h3>
<span class="paper-item__meta">arXiv / 2026-09-23 / Jonas Busk</span>
<p>用于先进材料应用（如电化学、催化和腐蚀）中固液界面的机器学习原子间势（MLIPs）需要能够同时采样液体环境、固体以及界面本身的训练数据。我们提出了 SoLiD26，一个经过整理的固液界面数据集，包含 1540 万个第一性原理原子结构，最多包含 576 个原子和 15 种化学元素，用于训练和评估 MLIPs。这些结构汇编自固液界面研究中进行的密度泛函理论（DFT）计算，大多数构型来源于从头算分子动力学（AIMD）模拟。每条记录包含原子种类、位置、模拟盒子、周期性边界条件、势能和原子力。SoLiD26 包括水系币金属界面、电极-电解质体系以及选定的体相参考结构，使用 VASP 并采用 PBE 泛函和 D3 色散校正计算。我们描述了用于构建该数据集的数据摄取和准备流程。通过在一组简单的训练、验证和测试划分上使用一系列 MACE 模型，展示了 SoLiD26 在训练和评估 MLIPs 中的应用。该数据集能够支持针对结构和化学上非均质的固液界面的 MLIPs 的开发与基准测试。</p>
<div class="paper-item__links"><a href="https://arxiv.org/pdf/2609.28013v1">PDF</a><a href="http://arxiv.org/abs/2609.28013v1">论文页面</a></div>
</article>
<article class="paper-item">
<h3>在检索和硬件约束下评估面向土耳其语领域文档的开放权重大型语言模型</h3>
<span class="paper-item__meta">arXiv / 2026-09-23 / Imtiaz Ul Hassan</span>
<p>大多数具备土耳其语能力的大语言模型（LLMs）是使用通用基准而非长篇、结构复杂的领域文档进行评估的。本文在资源受限的本地部署环境下，评估了五个开放权重7B-8B模型在土耳其语文档问答中的表现。主要基准包含100个经过系统验证的问题，这些问题源自一份109页的工业研发报告，评估协议通过第二份112页的公共部门报告和独立构建的100题集合进行了复现。所有模型均在配备6 GB显存的NVIDIA RTX 3050笔记本电脑GPU上本地评估，采用受控提示、解码和4比特量化。主要方法论贡献是一种证据标注评估协议，该协议无需额外模型调用即可将检索失败与下游模型推理失败区分开来。在主要基准上，端到端准确率范围为49%至75%。此外，使用95% Wilson区间和精确配对McNemar检验比较了七种词汇、稠密和混合检索配置；在任一文档上，均无配置显著优于字符TF-IDF基线。证据召回在两份报告中的饱和方式不同，表明检索和有效上下文容量对某些文档可能是约束条件，而对其他文档则不是。这些结果表明，在将开放权重LLMs部署于土耳其语领域文档时，必须分别评估模型选择、检索行为和硬件限制。</p>
<div class="paper-item__links"><a href="https://arxiv.org/pdf/2609.28007v1">PDF</a><a href="http://arxiv.org/abs/2609.28007v1">论文页面</a></div>
</article>
<article class="paper-item">
<h3>共享全局KV与层特定局部历史</h3>
<span class="paper-item__meta">arXiv / 2026-09-23 / Xinglang Xian</span>
<p>仅解码器的 Transformer 语言模型会缓存键和值（KV），以便在生成过程中复用过去的计算。跨层共享 KV 可以节省存储，但会降低跨深度可用表示的多样性。我们研究局部记忆应在共享全局 KV 之外保留什么，将历史内容与用于形成它的输入来源区分开来。在 1.26 亿参数和 2K 上下文下，一项八种子研究发现，使用局部历史比使用当前 token 的局部分支，留出测试困惑度约低 1.4%。容量、条目数和训练计算的控制支持历史内容的价值。在一项两种子比较中，当相邻层共享局部输入但保留独立投影时，这一价值仍然存在；来源共享还缩短了精确缓存构建依赖。与 GQA 和相邻层 KV 共享相比，在相同的有界学习率搜索和新种子确认下，使用更大的缓存和更高的长请求延迟，可以获得更好的同源似然。在等 token 适配到 8K 后，相对于相邻层共享的排序仍然存在，但伴随短上下文代价。八种子外部书籍历史效应仍不确定，下游结果因任务而异。我们推导出一个充分后缀调度，在精确算术中保留完整缓存的同时，减少上层构建工作。</p>
<div class="paper-item__links"><a href="https://arxiv.org/pdf/2609.28006v1">PDF</a><a href="http://arxiv.org/abs/2609.28006v1">论文页面</a></div>
</article>
<article class="paper-item">
<h3>疑问对话的受控属性特定摘要</h3>
<span class="paper-item__meta">arXiv / 2026-09-23 / A Aditya Bhardwaj</span>
<p>对疑问式对话进行有效摘要，是法证与调查场景中的一项关键任务，要求具备高事实准确性、连贯性以及特定属性相关性。在本研究中，我们提出了CASPER，一种用于评估式摘要的新型思维链属性特定提示框架，利用结构化提示与迭代优化来生成审讯者与证人互动的高质量摘要。我们构建了MINDSum，这是一个扩展MIND语料库的数据集，包含6，000个话语对，并标注了事件细节、事实陈述、人物描述和填充内容。CASPER采用RoleEval，一种分层评估机制，其中多个角色（警官、督察、高级督察）依据预定义标准对摘要进行迭代评估。通过整合实体抽取与结构化反馈循环，CASPER相较现有基线显著提升了事实一致性与上下文完整性。实验结果表明，我们的框架在词汇指标（ROUGE）和语义指标（BERTScore）上均优于标准摘要模型，同时人工评估也证实其与专家推理相一致。我们的发现凸显了受控摘要在高风险领域中的潜力，为AI驱动的法证情报铺平了道路。</p>
<div class="paper-item__links"><a href="https://arxiv.org/pdf/2609.28004v1">PDF</a><a href="http://arxiv.org/abs/2609.28004v1">论文页面</a></div>
</article>
<article class="paper-item">
<h3>从失败中学习：面向小型语言模型工具使用代理的异构图记忆</h3>
<span class="paper-item__meta">arXiv / 2026-09-23 / Jiaxing Li</span>
<p>中小型语言模型为工具使用型智能体提供了具有成本效益的执行器，使其在本地和大规模部署中具有吸引力。然而，在长时程和有状态环境中，它们常常犯结构性错误，例如遗漏必需的观察、执行过早写入、重复失败的调用以及违反动作前置条件。这些错误可能导致状态更新错误、策略违规以及代价高昂或不可逆的后果，使得可靠的工具执行成为一项关键部署挑战。现有的微调方法需要大量数据和计算，而扁平记忆可能检索到失败动作却无法保留其因果上下文或安全条件。在本文中，我们提出FRESH，一种基于经验结构化异构图结构的失败感知检索框架，它将历史成功与失败转化为工具使用型智能体的结构化外部经验。通过显式建模任务、动作、错误、修复和执行条件之间的依赖关系，FRESH帮助冻结的语言模型复用可靠策略、避免重复失败，并在有状态工具交互中做出更安全的决策。在$τ$-Bench和AppWorld上使用多个开源模型进行的实验表明，FRESH在任务成功率和工具使用可靠性方面持续优于无记忆智能体和具有代表性的基于记忆的基线方法。</p>
<div class="paper-item__links"><a href="https://arxiv.org/pdf/2609.28003v1">PDF</a><a href="http://arxiv.org/abs/2609.28003v1">论文页面</a></div>
</article>
<article class="paper-item">
<h3>LLM辅助的演化软件需求中结构差异可视化工作流</h3>
<span class="paper-item__meta">arXiv / 2026-09-23 / Koi McFarland</span>
<p>本文提出了一种用于可视化演化中软件需求结构差异的大语言模型辅助工作流。该工作流在OntologyWeb环境中实现，将基线和当前需求表示为基于三元组的语义图，并支持对精选图快照进行并排比较。比较视图对齐匹配的实体，并使用视觉编码来突出结构变化。</p>
<div class="paper-item__links"><a href="https://arxiv.org/pdf/2609.28002v1">PDF</a><a href="http://arxiv.org/abs/2609.28002v1">论文页面</a></div>
</article>
<article class="paper-item">
<h3>你的模型正在泄露：通过LLM残差流进行的隐蔽信息传递</h3>
<span class="paper-item__meta">arXiv / 2026-09-23 / Mingyuan Li</span>
<p>隐私敏感型组织可能在受限或气隙环境中运行大语言模型，同时导出选定的诊断产物。我们表明，一个被攻陷的运行时组件可以将敏感信息隐藏在允许离开受限环境的中间激活中。离线观察者可以用一个简单的线性解码器恢复该信息。该攻击不需要模型重训练或权重修改，不需要攻击者控制的外传通道，也不需要对记录器或传输过程拥有控制权。我们提出一种残差流隐蔽信道攻击，它将消息映射为码字，并通过一个被攻陷的运行时钩子将其注入中间残差流。为保持可恢复性，注入强度根据信噪残差范数比随局部残差范数缩放。在来自七个架构家族的十一个模型上，我们的评估显示，在九个模型上恢复率达到91--100%，KL散度为0.001--0.007，而所评估的激活级检测器仍接近随机猜测（AUC &lt;= 0.56）。所测试的事后防御无法可靠地消除该信道。因此，一个激活产物可以在模式上有效，同时携带未被授权跨越边界的信息。</p>
<div class="paper-item__links"><a href="https://arxiv.org/pdf/2609.27996v1">PDF</a><a href="http://arxiv.org/abs/2609.27996v1">论文页面</a></div>
</article>
<article class="paper-item">
<h3>合规于本地控制，集体歧视性。受监管金融中多智能体人工智能的治理架构</h3>
<span class="paper-item__meta">arXiv / 2026-09-23 / Jose Manuel de la Chica Rodriguez</span>
<p>金融机构已开始将代理式工作流部署于信贷、欺诈、催收、合规和运营控制领域。治理在很大程度上仍以组件为中心：每个模型或代理都在本地进行规范、测试、授权和监控。当机构风险源于众多本地可接受组件的联合行为时，这种做法是不够的。我们将这一缺口称为宪制性不可组合性：本地合规检查未必能组合成可接受的集体结果，例如有界差别影响、市场诚信或可追溯问责。我们提出ARIA作为面向金融领域的参考架构和可证伪的研究议程，用于代理群体治理。它在规范-问责、执行-控制和保证-学习三个平面上组织六项能力：策略规范、群体层面观测值与期望值行为监控（M2）、有界权限、运行时遏制、自适应策略变更，以及受保护的人类监督能力。两个模拟分别展示了在本地控制下共享信号薄档案排除，以及在构造的漂移机制中通过观测值与期望值分布监控实现的更早预警。该贡献将这些控制映射到公平贷款、欧盟人工智能法案、模型风险和 conduct-supervision 证据需求，并以验证议程收尾，而非声称生产有效性。</p>
<div class="paper-item__links"><a href="https://arxiv.org/pdf/2609.27994v1">PDF</a><a href="http://arxiv.org/abs/2609.27994v1">论文页面</a></div>
</article>
<article class="paper-item">
<h3>视觉Transformer特征空间的任务诱导黎曼度量</h3>
<span class="paper-item__meta">arXiv / 2026-09-23 / Andrew Bond</span>
<p>在Vision Transformer（ViT）特征空间上运行的方法通常依赖欧氏距离或余弦相似度。这假设每个方向都同等重要，但没有理由相信真实的任务几何具有这一性质。特征空间中任务敏感的几何由拉回度量 $g(F) = J(F)^\top J(F)$ 给出，其中 $J$ 是解码器输出（馈入任务特定距离）相对于特征的雅可比矩阵。在现代规模下存储完整的 $g$ 是不可行的，而对于深度图等稠密输出，即使构造 $J$ 也不切实际。我们表明，该度量的低秩近似是否可学习取决于模型-解码器对，并用一种无矩阵诊断量 $κ_{cap}(r)$ 来刻画，该诊断量可用少量雅可比向量积计算。对于可处理的配对，我们开发了谱拉回网络（SPN），它通过随机幂迭代学习度量的低秩版本，并将其蒸馏为一个 $310$K 参数的重要性头，直接从特征预测 token 重要性。当雅可比谱过于分散而无法进行低秩近似时，将解码器输入特征通过 VAE 瓶颈可以恢复可处理性。在 DPT、DINOv2、CLIP 和 VGGT 骨干网络上，$κ_{cap}(r)$ 能预测哪些学习度量架构可行。重要性头在 DINOv2 CLS 上达到 Spearman $ρ= 0.998$，并且我们的几何 token 剪枝在剪枝比例 $0.5$ 下将基于 ToMe 的 token 选择在 DPT 深度上的额外深度误差降低了 $25\%$，且无需微调 ViT。项目页面：https://cyberiada.github.io/TaskInducedViTs/</p>
<div class="paper-item__links"><a href="https://arxiv.org/pdf/2609.27988v1">PDF</a><a href="http://arxiv.org/abs/2609.27988v1">论文页面</a></div>
</article>
</div>
