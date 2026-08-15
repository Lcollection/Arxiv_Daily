# 最近更新

<div class="paper-overview">
<div class="paper-overview__item"><span>更新日期</span><strong><a href="../2026-08-15/">2026-08-15</a></strong></div>
<div class="paper-overview__item"><span>论文总数</span><strong>10</strong></div>
<div class="paper-overview__item"><span>数据接口</span><strong><a href="../../api/latest.json">latest.json</a></strong></div>
</div>

## 来源概览

<div class="paper-source-grid">
<div class="paper-source-card">
<span>arXiv</span>
<strong>10 篇</strong>
<a href="../2026-08-15-arxiv/">查看来源页面</a>
</div>
<div class="paper-source-card">
<span>bioRxiv</span>
<strong>0 篇</strong>
<a href="../2026-08-15-biorxiv/">查看来源页面</a>
</div>
<div class="paper-source-card">
<span>medRxiv</span>
<strong>0 篇</strong>
<a href="../2026-08-15-medrxiv/">查看来源页面</a>
</div>
</div>

## 当期论文

<div class="paper-list">
<article class="paper-item">
<h3>AutoDesign：面向长时程智能体设计的元驾驭优化</h3>
<span class="paper-item__meta">arXiv / 2026-08-13 / Yaxin Luo</span>
<p>将多模态源转化为精简且结构化的媒体输出，从根本上可视为一个以模型-框架系统为中心的长期智能体过程。理想的框架系统应契合人类设计先验，并通过经验探索积累可复用经验，以驱动递归式自我改进，然而现有范式仍属静态，未能达到此能力。本文提出AutoDesign框架，其与人类设计先验对齐，其中元框架优化器引导代码智能体基于回滚反馈递归改进框架。为实例化并评估该框架，我们聚焦于学术论文到海报生成任务，并引入PosterBench，包含覆盖五个学科的100篇论文主轨道，以及PosterBench-mini，一个用于受控评估的共享10篇论文子集。在PosterBench主轨道上，AutoDesign取得最高分78.32，超越闭源商业系统Claude Design 7.45分。在七种受控代码智能体-模型配置中，集成学习到的DesignHarness持续提升性能，将平均PosterBench分数从54.99提升至67.39（+12.4%）。在完全自主的长期循环中，它在40分钟内执行253次工具调用和11次编辑轮次，成本低于3美元，在人类评估中达到平均会议海报质量。一项系统盲法人类研究进一步表明，AutoDesign在评估系统中获得最高的人类偏好。</p>
<div class="paper-item__links"><a href="https://arxiv.org/pdf/2608.13560v1">PDF</a><a href="http://arxiv.org/abs/2608.13560v1">论文页面</a></div>
</article>
<article class="paper-item">
<h3>OmniScientist：一种全模态、全学科的人工智能科学家</h3>
<span class="paper-item__meta">arXiv / 2026-08-13 / Bobo Li</span>
<p>基础模型的最新进展使AI科学家能够自动化日益完整的研究工作流程，从假设生成和代码执行到稿件准备。然而，仅覆盖工作流程并不能提供科学发现所依赖的全部证据。现有系统通常基于文本、代码、标签或预计算摘要进行推理，使得科学上具有决定性的空间、时间、跨通道和程序性关系对智能体不可用。我们引入OmniScientist，一个端到端、全模态的AI科学家，直接从异构原始证据中进行多学科研究。一个感知层和3个自主智能体（分别负责构思、实验和撰写）在确定性流水线中运作，使观察能够在整个研究生命周期中塑造研究问题、实验决策和最终主张。通过在代码中运行想法、严谨性和主张检查，系统强制执行新颖性筛选、统计有效性、执行溯源和数值可追溯性。我们在涵盖5个学科家族、4类科学证据以及包括图像、信号、音频、视频、3D结构、轨迹、表格、公式和图在内的模态的36个真实数据案例上评估OmniScientist。系统在所有36个案例中完成了从原始数据到编译稿件的完整路径，并在参考推理骨干下实现了平均总体论文得分6.3。在与仅接收预计算标量特征的盲变体的配对比较中，直接感知改善了所有7个评估维度，并在85%的正面比较判断中获胜。这些结果表明，生命周期范围内的感知对于基于证据的科学发现至关重要，并为实现广泛能力的AI科学家提供了实用路径。</p>
<div class="paper-item__links"><a href="https://arxiv.org/pdf/2608.13558v1">PDF</a><a href="http://arxiv.org/abs/2608.13558v1">论文页面</a></div>
</article>
<article class="paper-item">
<h3>V-RAE：重新思考用于生成的视频潜在空间</h3>
<span class="paper-item__meta">arXiv / 2026-08-13 / Minghui Guo</span>
<p>潜在视频生成依赖于自编码器来定义一个紧凑空间，生成模型在其中运作。尽管视频自编码器架构已大幅演进，其潜在空间仍主要针对像素级重建优化，缺乏高层语义组织。然而，重建最优的潜在空间未必适合生成建模。我们提出V-RAE，一种视频表示自编码器，在冻结的视觉基础模型表示之上构建紧凑的生成潜在空间。一个轻量级时间池化模块去除时间冗余，同时保留语义结构，视频解码器从压缩特征中重建连续运动。我们使用四种代表性的冻结编码器评估V-RAE，涵盖视频重建、语义探测和类别条件生成。V-RAE在K600上达到2.13 rFVD，优于所有评估的大规模预训练视频VAE。其潜在空间比传统视频分词器潜在空间保留显著更多的语义信息。在匹配的生成设置下，我们最佳变体在UCF101和K600上分别达到117.86和19.16的gFVD分数，同时收敛速度提升高达6倍。我们进一步表明，仅重建质量不足以表征生成效用，并引入tFVD，一种时间一致性诊断指标，与下游生成质量更可靠地相关。除视频生成外，V-RAE在匹配预测设置下，在Cityscapes上的未来视频预测也优于Wan 2.2 VAE潜在空间。综合实验表明，冻结的语义表示能支持视频重建、生成和预测建模。项目页面：https://v-rae.github.io/。</p>
<div class="paper-item__links"><a href="https://arxiv.org/pdf/2608.13556v1">PDF</a><a href="http://arxiv.org/abs/2608.13556v1">论文页面</a></div>
</article>
<article class="paper-item">
<h3>HumanTracker：迈向全面且与人类对齐的运动追踪基准</h3>
<span class="paper-item__meta">arXiv / 2026-08-13 / Dairu Liu</span>
<p>人形运动跟踪是遥操作和全身模仿的核心，但评估往往与人们在视频中感知到的内容不一致。运动学误差平均每帧姿态差异，却忽略了最关键的物理伪影，尤其是支撑不稳定和错误接触，如脚部滑动和触地时机不当。同时，广泛使用的测试套件规模较小，缺乏应对接触密集、长时程行为所需的多样性。我们引入HumanTracker，使人形跟踪评估既感知对齐又可扩展。HumanTracker基准包含来自多位专业表演者约153小时的光学运动轨迹，组织为四个运动家族，并附有文本标签以支持细粒度诊断。我们进一步提出HumanScore，一种基于偏好对齐的度量，在包含24K运动的12K运动对上训练。在代表性最先进跟踪器上，HumanScore更准确地预测人类偏好，并揭示运动学度量常遗漏的接触和稳定性失败。</p>
<div class="paper-item__links"><a href="https://arxiv.org/pdf/2608.13555v1">PDF</a><a href="http://arxiv.org/abs/2608.13555v1">论文页面</a></div>
</article>
<article class="paper-item">
<h3>在线概率预测的防御性提升</h3>
<span class="paper-item__meta">arXiv / 2026-08-13 / Georgy Noarov</span>
<p>我们研究由自适应对手选择的二元结果的在线概率预测。给定一个针对弱假设类$H$的在线学习算法，我们希望高效地获得两种不可比较的保证，这些保证是现有在线提升技术分别提供的。在线梯度提升在Brier分数上与$H$的跨度在每条序列上诱导的最佳预测器竞争，但当跨度不包含准确预测器时则无任何承诺。在线弱到强提升在弱学习条件下将分类误差驱动至零，但当该条件不成立时承诺甚少。我们给出一个简单的防御性预测算法，即防御性提升器，它同时获得这两种保证。在每条自适应序列上，其Brier分数与$H$跨度诱导的最佳预测竞争，速率与在线梯度提升相同；同时，每当实现的转录满足平滑弱学习条件时，其Brier分数和随机分类误差满足与在线分类提升相同的速率保证。这是通过操作提升的“对偶视角”实现的：当算法的随机分类误差持续较高时，其错误权重形成一个平滑重加权，在该重加权下每个弱假设具有低边缘，从而产生一个事后硬核证书，表明弱学习条件不成立。我们还开发了一个强自适应变体，它在每个时间间隔上同时满足两种保证。防御性提升器非常高效：它仅访问一个弱类学习器，而我们比较的先前在线提升方法则维护大型弱学习器集成。在合成和真实数据流上的实验展示了其强大的预测性能（有时显著优于所有先前基线），同时运行速度快了几个数量级。</p>
<div class="paper-item__links"><a href="https://arxiv.org/pdf/2608.13554v1">PDF</a><a href="http://arxiv.org/abs/2608.13554v1">论文页面</a></div>
</article>
<article class="paper-item">
<h3>PlayWorld：基于智能体玩家在长时程目标上对世界模型进行基准测试</h3>
<span class="paper-item__meta">arXiv / 2026-08-13 / Kaixin Ding</span>
<p>视频世界模型根据当前观测和用户动作模拟未来状态。近期系统在长序列中展示了令人印象深刻的视频一致性和动作可控性。然而，公平比较这些交互式模型仍具挑战性。实践中，人类玩家通常通过追求长期目标来评估世界模型，例如，用户可能旋转360度以检查环境是否保持一致，或走入水中观察是否生成逼真的水波纹。实现相同目标所需的动作序列在不同模型间可能差异显著，这使得固定动作条件评估不适合跨模型比较。为解决此问题，我们采用多模态智能体玩家与模型交互，以达成指定的长期目标。基于此范式，我们引入PlayWorld基准，提供171个场景，每个场景设有明确目标。为全面评估性能，我们从四个核心维度衡量模型：几何一致性、交互保真度、视野外演化及洞察演化。此外，我们纳入视频质量和可控性的基础能力指标。对九个最先进世界模型的实验表明，当前模型在长期交互目标上仍不可靠，尤其在维持空间一致性和持续状态演化方面。代码和数据可在https://github.com/kxding/PlayWorld获取。</p>
<div class="paper-item__links"><a href="https://arxiv.org/pdf/2608.13552v1">PDF</a><a href="http://arxiv.org/abs/2608.13552v1">论文页面</a></div>
</article>
<article class="paper-item">
<h3>多标签Jaccard度量的指数凸校准维度</h3>
<span class="paper-item__meta">arXiv / 2026-08-13 / Mingyuan Zhang</span>
<p>逐实例Jaccard分数，即交并比（IoU），在多标签分类和二元分割中是标准指标。对于$s$个标签，其损失矩阵有$2^s$个结果和报告。在约定$\mathrm{Jac}(\varnothing,\varnothing)=1$下，我们证明Jaccard分数、平移损失和普通损失矩阵是非奇异的，且损失列的仿射维度为$2^s-1$。证明结合了有限MinHash Gram表示与布尔Möbius反演。对于精确校准，我们证明$2^{s-1} \leq \mathrm{CCdim}(L^{\mathrm{Jac}}) \leq 2^s-1$。下界使用一个因子加权分布，具有$2^{s-1}+1$个支持结果和贝叶斯最优报告。因此，每个精确校准的凸替代需要指数多个预测坐标。我们还给出了两个多项式维度的近似保证，并带有显式遗憾转移。一个新的$F_1$-到-Jaccard转移将现有的$(s^2+1)$维$F_1$替代转化为一个多项式时间规则，其渐近Jaccard遗憾至多为$3-2\sqrt{2}$。对于任意$α&gt;0$和$0&lt;ρ&lt;1$，一个MinHash平方损失替代在任意条件标签分布上均匀达到Jaccard遗憾下限$α$。以至少$1-ρ$的概率，直接构造的维度为$O((s^2+s\log(1/ρ))/α^2)$，而带符号变体的维度为$O((s+\log(1/ρ))/α^2)$。因此，零遗憾校准需要指数维度，而每个固定的加性遗憾容差允许多项式预测维度。</p>
<div class="paper-item__links"><a href="https://arxiv.org/pdf/2608.13549v1">PDF</a><a href="http://arxiv.org/abs/2608.13549v1">论文页面</a></div>
</article>
<article class="paper-item">
<h3>QuoteBench：匹配分数如何掩盖命令路径故障</h3>
<span class="paper-item__meta">arXiv / 2026-08-13 / Shangao Li</span>
<p>LLM编码代理通过可能序列化、包装并重新解析模型输出的接口发出Bash命令。仅凭匹配执行分数无法区分命令生成错误与生成后引入的失败。QuoteBench通过56个来自14个事件衍生任务族的单次任务上的精确最终状态验证来测量这一边界，将生成契约与执行传输交叉在一个故意未转义的附加解析器上。在插值点转义可重现每个重放回复的原始路径结果，因此在已披露边界下的任何恢复必须来自模型改变其生成。在八个同窗口配置中，通过附加解析器重放相同回复将成功率降低55.4至73.2个百分点；披露为六个配置恢复30.4至60.7个百分点，另外两个配置恢复为零或略负。原始生成在前沿几乎饱和；边界适应才是区分模型的要素。GPT-5.6-sol的匹配差距为-3.6个百分点，隐藏了-64.3个百分点的损害和+60.7个百分点的补偿。部署配置重新排序模型：26个可比对中有一个反转明确无误，另有四个位于单任务边缘。对命令发出代理的评估应报告模型配置、生成契约、执行路径、操作点和最终状态验证器，而非将匹配分数视为模型的内在属性。</p>
<div class="paper-item__links"><a href="https://arxiv.org/pdf/2608.13547v1">PDF</a><a href="http://arxiv.org/abs/2608.13547v1">论文页面</a></div>
</article>
<article class="paper-item">
<h3>Alaya-EVOKE：从线性扩展监督到无尽世界</h3>
<span class="paper-item__meta">arXiv / 2026-08-13 / Yuanyang Yin</span>
<p>交互式世界模型必须支持持久记忆、响应式交互和长时程生成，然而这些需求对模型提出了相互冲突的要求。在去噪器上下文或键值缓存中维护历史会带来不断增长的成本，迫使在会话长度和保留记忆之间进行权衡，而低延迟交互依赖于少步生成，其能力受限于其教师模型。Evoke通过外部化持久世界状态并重新设计教师模型以支持长时程交互生成，解决了这两个限制。场景几何信息维护在一个外部的、以相机为索引的世界状态库中，仅检索与视图相关的信息，从而在会话增长时保持去噪器上下文有界。我们不将教师模型视为固定生成器，而是将其设计用于长时程监督：其稀疏注意力结合了分块分组、检索选定的远距离帧以及线性注意力全局状态，在实现长时程监督的同时，内存和计算开销呈线性增长。这种监督暴露了在短窗口内局部看似合理的内容漂移，而逐块条件化则允许在整个序列中进行提示更改和事件控制。一个30秒的分布匹配目标，在自强制展开下应用，将这两种能力迁移到不使用无分类器引导的三步学生模型中，提高了对长期漂移的抵抗力，同时保持了响应式条件化。凭借有界上下文和循环外部记忆，Evoke支持开放式、持续演化的生成；在单个H200上，分辨率为$384\times 640$时，每个$1.5\,\mathrm{s}$的块在$2.11\,\mathrm{s}$内生成。作为三步世界模型，Evoke在WBench上达到了最先进的性能，同时在VBench-Long和VBench-2.0上保持竞争力。</p>
<div class="paper-item__links"><a href="https://arxiv.org/pdf/2608.13546v1">PDF</a><a href="http://arxiv.org/abs/2608.13546v1">论文页面</a></div>
</article>
<article class="paper-item">
<h3>LittleLearner：教学控制知识暴露下的语言模型</h3>
<span class="paper-item__meta">arXiv / 2026-08-13 / Fanfei Li</span>
<p>现代语言模型是在异构的网络规模文本语料库上训练的。因此，研究知识和技能的获取变得困难，因为先前接触相关内容的情况难以刻画。为应对这一挑战，我们引入了LITTLECURRICULUM，这是一个精心策划的880亿词元预训练语料库，专门针对美国小学教材内容，明确排除了五年级以上教授的概念、事实和词汇。在LITTLECURRICULUM上从头训练一个50亿参数的LLM，得到LITTLELEARNER，该模型具备足够的语言能力以进行开放式评估，但其知识和能力边界清晰，映射到可解释的课程指南。我们发布LITTLECURRICULUM和LITTLELEARNER，作为一个发展受限的沙盒，用于研究模型在明确定义的训练范围内如何获取、表示和使用数据。我们通过一系列初步实验展示了该沙盒的实用性，这些实验涉及通过后训练和上下文学习注入新知识。这些方法使LITTLELEARNER能更好地利用现有知识，但并未提升超出范围的能力。我们的发现强调了这一受控环境对未来研究的重要价值。</p>
<div class="paper-item__links"><a href="https://arxiv.org/pdf/2608.13545v1">PDF</a><a href="http://arxiv.org/abs/2608.13545v1">论文页面</a></div>
</article>
</div>
